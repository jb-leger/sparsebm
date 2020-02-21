import numpy as np
import scipy as sp
import scipy.sparse


def generate_bernouilli_SBM(n, nq, pi, alpha, symetric=False):
    print("Start generating graph, it might take a while...")
    Y = np.random.multinomial(1, alpha.flatten(), size=n)
    classes = [Y[:, q].nonzero()[0] for q in range(nq)]

    inserted = set()
    for q in range(nq):
        for l in range(nq):
            if pi[q, l] >= 0.25:
                # rejection algo not effecient
                for i in classes[q]:
                    nb_ones = np.random.binomial(classes[l].size, pi[q, l])
                    trucs = np.random.choice(classes[l], nb_ones, replace=False)
                    inserted.update((i, j) for j in trucs)
            else:
                nb_ones = np.random.binomial(
                    classes[q].size * classes[l].size, pi[q, l]
                )
                c = 0
                while c < nb_ones:
                    i = np.random.choice(classes[q])
                    j = np.random.choice(classes[l])
                    if (i, j) not in inserted:
                        inserted.add((i, j))
                        c += 1
    if symetric:
        inserted = [(i, j) for (i, j) in inserted if i < j]
        inserted.extend([(j, i) for (i, j) in inserted])
    else:
        inserted = [(i, j) for (i, j) in inserted if i != j]
    X = sp.sparse.coo_matrix(
        (np.ones(len(inserted)), ([i for i, j in inserted], [j for i, j in inserted])),
        (n, n),
    )

    return {"X": X, "Y": Y, "alpha": alpha, "pi": pi}


def generate_bernouilli_LBM(n1, n2, nq, nl, pi, alpha_1, alpha_2):
    print("Start generating graph, it might take a while...")
    Y1 = np.random.multinomial(1, alpha_1.flatten(), size=n1)
    Y2 = np.random.multinomial(1, alpha_2.flatten(), size=n2)
    row_classes = [Y1[:, q].nonzero()[0] for q in range(nq)]
    col_classes = [Y2[:, l].nonzero()[0] for l in range(nl)]

    inserted = set()
    for q in range(nq):
        print(q)
        for l in range(nl):
            if pi[q, l] >= 0.25:
                # rejection algo not effecient
                for i in row_classes[q]:
                    nb_ones = np.random.binomial(col_classes[l].size, pi[q, l])
                    trucs = np.random.choice(col_classes[l], nb_ones, replace=False)
                    inserted.update((i, j) for j in trucs)
            else:
                nb_ones = np.random.binomial(
                    row_classes[q].size * col_classes[l].size, pi[q, l]
                )
                c = 0
                while c < nb_ones:
                    i = np.random.choice(row_classes[q])
                    j = np.random.choice(col_classes[l])
                    if (i, j) not in inserted:
                        inserted.add((i, j))
                        c += 1

    X = sp.sparse.coo_matrix(
        (np.ones(len(inserted)), ([i for i, j in inserted], [j for i, j in inserted])),
        (n1, n2),
    )

    return {
        "X": X,
        "Y1": Y1,
        "Y2": Y2,
        "alpha_1": alpha_1,
        "alpha_2": alpha_2,
        "pi": pi,
    }
