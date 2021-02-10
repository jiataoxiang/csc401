import numpy as np
if __name__ == "__main__":
    out1 = np.load("./test.npz")["arr_0"]
    out2 = np.load("./output/sample_feats.npz")["arr_0"]
    x, y = out1.shape
    print("number of samples: ", x, "\nnumber of columns: ", y, "\n")
    for i in range(x):
        for j in range(y):
            if out1[i][j] != out2[i][j]:
                print(i+1, j+1, ",", "out1: ", out1[i][j], ", out2: ", out2[i][j], "\n")
