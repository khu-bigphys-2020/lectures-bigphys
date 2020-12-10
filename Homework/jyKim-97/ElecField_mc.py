import numpy as np
import matplotlib.pyplot as plt
import tqdm
import abc


def get_elecfield(pos, rhos, target_pos, vol=1, k=1):
    # rhos (n,)
    # pos, (ndim x n)
    # target_pos, (ndim x m)
    ndim = pos.shape[0]
    Es = np.zeros(target_pos.shape)
    # calculate distance
    for i in range(target_pos.shape[1]):
        if any(np.isnan(target_pos[:, i])):
            Es[:, i] = np.nan
            continue
        dr = target_pos[:, np.newaxis, i] - pos
        r3 = abs(np.sum(dr**2, axis=0))**(3/2)
        for dim in range(ndim):
            Es[dim, i] = np.sum(k * rhos * dr[dim, :] / r3)
    Es *= vol
    return Es


def get_potfield(pos, rhos, target_pos, vol=1, k=1):
    # rhos (n,)
    # pos, (ndim x n)
    # target_pos, (ndim x m)
    ndim = pos.shape[0]
    Vs = np.zeros(target_pos.shape[1])
    # calculate distance
    for i in range(target_pos.shape[1]):
        if any(np.isnan(target_pos[:, i])):
            Vs[i] = np.nan
            continue
        dr = target_pos[:, np.newaxis, i] - pos
        r = np.sqrt(np.sum(dr**2, axis=0))
        for dim in range(ndim):
            Vs[i] = np.sum(k * rhos / r)
    Vs *= vol
    return Vs


class mcSampling:
    __metaclass__ = abc.ABCMeta
    def __init__(self, slim, ndim=3, seed=None):
        # slim, (ndim, 2)
        self.slim = slim
        self.ndim = ndim
        np.random.seed(seed)

    def get_rand_pts(self, itr=int(1e6)):
        pos = np.zeros([self.ndim, itr])
        rho = np.zeros(itr)
        for i in tqdm.tqdm(range(itr), ncols=100):
            pos[:, i] = self.rand_gen()
            rho[i] = self.frho(pos[:, i])
        return pos, rho

    def rand_gen(self):
        # pick random variable satisfy fboundary
        is_in = False
        while not is_in:
            r = []
            for i in range(self.ndim):
                r.append(np.random.uniform(low=self.slim[i][0], high=self.slim[i][1]))
            r = np.array(r)
            is_in = self.fboundary(r)
        return r
    
    @abc.abstractmethod
    def fboundary(self, r):
        # boundary function, return True / False
        pass

    @abc.abstractmethod
    def frho(self, r):
        # charge density function
        pass


if __name__ == '__main__': # for test

    class TestSample(mcSampling):
        def fboundary(self, r):
            return np.sqrt(sum(r**2)) <= (3/(4*np.pi))**(1/3)
        
        def frho(self, r):
            return 1

    # get Sample pts
    seed = 100010
    rmax = (3/(4*np.pi))**(1/3)
    rlim = [[-rmax, rmax], [-rmax, rmax], [-rmax, rmax]]
    itr = 1e5
    vol = 4/3*np.pi*rmax**3 / itr
    target_pos = np.array([10, 0, 0]).reshape(3, 1)

    mc_obj = TestSample(rlim, seed=seed)
    pts, rho = mc_obj.get_rand_pts(itr=int(itr))
    # calculate electric field
    es = get_elecfield(pts, rho, target_pos, vol=vol, k=1)
    # print result
    print("F = (%f, %f, %f)"%(es[0, 0], es[1, 0], es[2, 0]))
    print("|F| = %f"%(np.sqrt(np.sum(es**2))))
