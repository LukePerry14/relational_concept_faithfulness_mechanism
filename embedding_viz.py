import numpy as np

try:
    import plotly.express as px
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except Exception:
    PLOTLY_AVAILABLE = False
    import matplotlib.pyplot as plt


def cosine_similarity_matrix(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    Xn = X / norms
    return float_round(np.dot(Xn, Xn.T))


def float_round(A):
    try:
        return np.asarray(A, dtype=float)
    except Exception:
        return np.array(A)


def euclidean_distance_matrix(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    diffs = X[:, None, :] - X[None, :, :]
    d2 = np.sum(diffs * diffs, axis=2)
    return np.sqrt(np.maximum(d2, 0.0))


def pca_2d(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    Xc = X - np.mean(X, axis=0, keepdims=True)
    # SVD-based PCA
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    comps = Xc @ Vt.T
    return comps[:, :2]


def compare_embeddings(embeddings: list, labels: list | None = None, show_plots: bool = True):
    """
    embeddings: sequence of 1D numpy arrays (same dimensionality)
    labels: optional list of labels

    Prints pairwise cosine / euclidean matrices, lists top/bottom pairs,
    and shows a 2D PCA scatter + cosine heatmap (interactive if plotly available).
    """
    X = np.stack([np.asarray(e, dtype=float) for e in embeddings], axis=0)
    n = X.shape[0]
    if labels is None:
        labels = [str(i) for i in range(n)]

    cosm = cosine_similarity_matrix(X)
    eucm = euclidean_distance_matrix(X)

    print("Pairwise cosine similarity:")
    print(np.round(cosm, 4))
    print("\nPairwise Euclidean distance:")
    print(np.round(eucm, 4))

    # Top similar pairs (excluding diagonal)
    if n > 1:
        idxs = np.triu_indices(n, k=1)
        pairs = list(zip(idxs[0], idxs[1]))
        cos_vals = [cosm[i, j] for i, j in pairs]
        eu_vals = [eucm[i, j] for i, j in pairs]

        # sort by cosine
        sorted_by_cos = sorted(zip(pairs, cos_vals, eu_vals), key=lambda v: -v[1])
        print("\nTop pairs by cosine similarity:")
        for (i, j), cval, eval_ in sorted_by_cos[:min(5, len(sorted_by_cos))]:
            print(f"  {labels[i]} <-> {labels[j]}: cosine={cval:.4f}, euclid={eval_:.4f}")

    # 2D PCA scatter
    pts = pca_2d(X)

    if show_plots:
        if PLOTLY_AVAILABLE:
            fig = px.scatter(x=pts[:, 0], y=pts[:, 1], text=labels, title="PCA (2D) of embeddings")
            fig.update_traces(textposition="top center")
            fig.show()

            heat = go.Figure(data=go.Heatmap(z=cosm, x=labels, y=labels, colorscale="RdBu", zmin=-1, zmax=1))
            heat.update_layout(title="Cosine similarity matrix")
            heat.show()
        else:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.scatter(pts[:, 0], pts[:, 1])
            for i, lbl in enumerate(labels):
                ax.text(pts[i, 0], pts[i, 1], lbl)
            ax.set_title("PCA (2D) of embeddings")
            plt.show()

            plt.figure()
            plt.imshow(cosm, cmap="RdBu", vmin=-1, vmax=1)
            plt.colorbar()
            plt.xticks(range(n), labels, rotation=45)
            plt.yticks(range(n), labels)
            plt.title("Cosine similarity matrix")
            plt.tight_layout()
            plt.show()

    return {"cosine": cosm, "euclidean": eucm, "pca2d": pts}


def compare_metapath_feature_embeddings(paths):
    unique_embeddings = []
    unique_labels = []
    seen_vectors = set()

    # 2. Extract unique node embeddings from the MetaPaths
    for path in paths:
        for i, feat in enumerate(path.node_features):
            # Create a hashable version to check for uniqueness
            feat_id = tuple(feat.round(5).flatten()) 
            if feat_id not in seen_vectors:
                unique_embeddings.append(feat)
                # Label with node type and path source for context
                unique_labels.append(f"{path.node_types[i]} ({path.path_name})")
                seen_vectors.add(feat_id)

    compare_embeddings(unique_embeddings, unique_labels)