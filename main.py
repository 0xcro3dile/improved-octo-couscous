import numpy as np
import one_class

if __name__ == '__main__':


    X = np.array([[0,0,1],
                  [0,1,1],
                  [1,0,1],
                  [1,1,1]])

    y = np.array([[0],[1],[1],[0]])
    
    nn = NerualNetworks(X,y)

    for i in range(1500):
        nn.FeedFoward()
        nn.BackProp()

    print(nn.output)


