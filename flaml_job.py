
import os, sys, json, pickle, time, warnings, resource
import numpy as np, pandas as pd

def _vmsize():
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmSize:"):
                return int(line.split()[1]) * 1024

def job(a):
    warnings.filterwarnings("ignore")
    soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    limit = _vmsize() + a["mem_limit_mb"] * 1024 ** 2
    resource.setrlimit(resource.RLIMIT_AS, (limit if hard == resource.RLIM_INFINITY else min(limit, hard), hard))
    out = dict(status="ok", err="")
    try:
        from flaml import AutoML
        import shap

        def frame(X):
            df = pd.DataFrame(np.asarray(X, dtype=float), columns=a["feature_names"])
            for j, c in enumerate(a["feature_names"]):
                if a["cat_mask"][j]:
                    df[c] = pd.Categorical(df[c].astype(int), categories=list(range(a["n_levels"][j])))
            return df

        def fit(Xn, yn, seed):
            kw = dict(X_train=frame(Xn), y_train=yn, task="classification", metric="log_loss",
                      estimator_list=[a["learner"]], starting_points={a["learner"]: a["config"]},
                      max_iter=1, retrain_full=True, seed=seed, n_jobs=1, verbose=0)
            if a["balancing"] == "balanced":
                w = len(yn) / (2.0 * np.bincount(yn, minlength=2))
                kw["sample_weight"] = w[yn]
            aml = AutoML()
            aml.fit(**kw)
            return aml

        t0 = time.time()
        aml = fit(a["X_train"], a["y_train"], a["seed"])
        out["fit_secs"] = time.time() - t0
        out["config_match"] = bool(all(aml.best_config.get(k) == v for k, v in a["config"].items()))

        def g(X):
            X = np.asarray(X, dtype=float)
            return np.concatenate([aml.predict_proba(frame(X[i:i + 4096]))[:, 1] for i in range(0, len(X), 4096)])

        masker = shap.maskers.Independent(a["B"], max_samples=len(a["B"]))
        np.random.seed(a["perm_seed"])
        ex = shap.PermutationExplainer(g, masker, seed=a["perm_seed"])
        np.random.seed(a["perm_seed"])
        t1 = time.time()
        e = ex(a["XE"], max_evals=a["max_evals"], silent=True)
        out["explain_secs"] = time.time() - t1
        phi = np.asarray(e.values); base = np.asarray(e.base_values).reshape(-1)
        pE = g(a["XE"])
        out.update(m=np.abs(phi).mean(0), pE=pE, efficiency_error=float(np.abs(phi.sum(1) + base - pE).max()))

        if a["do_cv"]:                                         # 5-fold CV on the fixed folds
            oof = np.empty(len(a["y_train"]))
            for fit_pos, val_pos in a["cv_folds"]:
                m_ = fit(a["X_train"][fit_pos], a["y_train"][fit_pos], a["seed"])
                oof[val_pos] = m_.predict_proba(frame(a["X_train"][val_pos]))[:, 1]
            out["oof"] = oof
    except MemoryError:
        out.update(status="memout")
    except Exception as ex_:
        out.update(status="error", err=f"{type(ex_).__name__}: {str(ex_)[:300]}")
    tmp = a["out_path"] + ".tmp.npz"
    np.savez(tmp, **out)
    os.replace(tmp, a["out_path"])

def run_isolated(a, time_limit_s, workdir):
    import subprocess, tempfile
    fd, apath = tempfile.mkstemp(suffix=".pkl", dir=a["tmp_dir"]); os.close(fd)
    with open(apath, "wb") as f:
        pickle.dump(a, f, protocol=4)
    t0, status, err = time.time(), None, ""
    try:
        r = subprocess.run([sys.executable, "-m", "flaml_job", apath], cwd=workdir, timeout=time_limit_s,
                           stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        if r.returncode != 0:
            status, err = "crashed", f"exit {r.returncode}; " + r.stderr.decode(errors="ignore")[-300:]
    except subprocess.TimeoutExpired:
        status, err = "timeout", f"exceeded {time_limit_s}s"
    finally:
        os.remove(apath)
    if os.path.exists(a["out_path"]):
        return "done"
    np.savez(a["out_path"], status=status or "crashed", err=err, total_secs=time.time() - t0)
    return status or "crashed"

if __name__ == "__main__":
    with open(sys.argv[1], "rb") as f:
        job(pickle.load(f))
