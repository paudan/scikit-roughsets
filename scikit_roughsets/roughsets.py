import numpy as np

class RoughSetsReducer:

    def __size(self, x):
        return (1, x.shape[0]) if x.ndim == 1 else x.shape

    '''
    Calculates indiscernibility relation
    '''
    def indisc(self, a, x):

        def codea(a, x, b):
            yy = 0
            for i in range(0, a):
                yy += x[i] * b**(a-(i+1))
            return yy

        p, q = self.__size(x)
        ap, aq = self.__size(a)
        z = [e for e in range(1, q+1)]
        tt = np.setdiff1d(z, a)
        tt_ind = np.setdiff1d(z, tt)-1
        if x.ndim == 1:
            x = x[tt_ind]
        else:
            x = x[:, tt_ind]
        y = x
        v = [codea(aq, y, 10) for i in range(0, p)] if y.ndim == 1 \
            else [codea(aq, y[i, :], 10) for i in range(0, p)]
        y = np.transpose(v)
        if y.shape[0] == 1 and len(y.shape) == 1:
            I, yy = [1], [y]
            y = np.hstack((y, I))
            b, k, l = [y], [1], [1]
        else:
            ax = 1 if y.ndim > 1 else 0
            yy = np.sort(y, axis=ax)
            I = y.argsort(axis=ax)
            y = np.hstack((yy, I))
            b, k, l = np.unique(yy, return_index=True, return_inverse=True)
        y = np.hstack((l, I))
        m = np.max(l)
        aa = np.zeros((m+1, p), dtype=int)
        for ii in range(0, m+1):
            for j in range(0, p):
                if l[j] == ii:
                    aa[ii, j] = I[j]+1
        return aa

    '''
    Calculates lower approximation set of y
    '''
    def rslower(self, y, a, T):
        z = self.indisc(a, T)
        w = []
        p, q = self.__size(z)
        for u in range(0, p):
            zz = np.setdiff1d(z[u, :], 0)
            if np.in1d(zz, y).all():
                w = np.hstack((w, zz))
        return w.astype(dtype=int)

    '''
    Calculates upper approximation set of y
    '''
    def rsupper(self, y, a, T):
        z = self.indisc(a, T)
        w = []
        p, q = self.__size(z)
        for u in range(0, p):
            zz = np.setdiff1d(z[u, :], 0)
            zzz = np.intersect1d(zz, y)
            if len(zzz) > 0:
                w = np.hstack((w, zz))
        return w.astype(dtype=int)


    def __pospq(self, p, q):
        pm, pn = self.__size(p)
        qm, qn = self.__size(q)
        num = 0
        pp, qq = [[]] * pm, [[]] * qm
        for i in range(0, pm):
            pp[i] = np.unique(p[i, :])
        for j in range(0, qm):
            qq[j] = np.unique(q[j, :])
        b = []
        for i in range(0, qm):
            for j in range(0, pm):
                if np.in1d(pp[j], qq[i]).all():
                    num += 1
                    b = np.hstack((b, pp[j]))
        bb = np.unique(b)
        if bb.size == 0:
            dd = 1
        else:
            _, dd = self.__size(bb)
        y = float(dd - 1)/pn if 0 in bb else float(dd)/pn
        b = np.setdiff1d(bb, 0)
        return y, b

    '''
    Extract core set from C to D
    '''
    def core(self, C, D):
        x = np.hstack((C, D))
        c = np.array(range(1, C.shape[1]+1))
        d = np.array([C.shape[1]+1])
        cp, cq = self.__size(c)
        q = self.indisc(d, x)
        pp = self.indisc(c, x)
        b, w = self.__pospq(pp, q)
        a, k, kk, p = ([[]] * cq for i in range(4))
        y = []
        for u in range(0, cq):
            ind = u+1
            a[u] = np.setdiff1d(c, ind)
            p[u] = self.indisc(a[u], x)
            k[u], kk[u] = self.__pospq(p[u], q)
            if k[u] != b:
                y = np.hstack((y, ind))
        return np.array(y)

    def __sgf(self, a, r, d, x):
        pr = self.indisc(r, x)
        q = self.indisc(d, x)
        b = np.hstack((r, a))
        pb = self.indisc(b, x)
        p1, _ = self.__pospq(pb, q)
        p2, _ = self.__pospq(pr, q)
        return p1 - p2

    '''
    Return the set of irreducible attributes
    '''
    def reduce(self, C, D):

        def redu2(i, re, c, d, x):
            yre = re
            re1, re2 = self.__size(re)
            q = self.indisc(d, x)
            p = self.indisc(c, x)
            pos_cd, _ = self.__pospq(p, q)
            y, j = None, None
            for qi in range(i, re2):
                re = np.setdiff1d(re, re[qi])
                red = self.indisc(re, x)
                pos_red, _ = self.__pospq(red, q)
                if np.array_equal(pos_cd, pos_red):
                    y = re
                    j = i
                    break
                else:
                    y = yre
                    j = i + 1
                    break
            return y, j

        x = np.hstack((C, D))
        c = np.array(range(1, C.shape[1]+1))
        d = np.array([C.shape[1]+1])
        y = self.core(C, D)
        q = self.indisc(d, x)
        p = self.indisc(c, x)
        pos_cd, _ = self.__pospq(p, q)
        re = y
        red = self.indisc(y, x)
        pos_red, _ = self.__pospq(red, q)
        while pos_cd != pos_red:
            cc = np.setdiff1d(c, re)
            c1, c2 = self.__size(cc)
            yy = [0] * c2
            for i in range(0, c2):
                yy[i] = self.__sgf(cc[i], re, d, x)
            cd = np.setdiff1d(c, y)
            d1, d2 = self.__size(cd)
            for i in range(d2, c2, -1):
                yy[i] = []
            ii = np.argsort(yy)
            for v1 in range(c2-1, -1, -1):
                v2 = ii[v1]
                re = np.hstack((re, cc[v2]))
                red = self.indisc(re, x)
                pos_red, _ = self.__pospq(red, q)
        re1, re2 = self.__size(re)
        core = y
        for qi in range(re2-1, -1, -1):
            if re[qi] in core:
                y = re
                break
            re = np.setdiff1d(re, re[qi])
            red = self.indisc(re, x)
            pos_red, _ = self.__pospq(red, q)
            if np.array_equal(pos_cd, pos_red):
                y = re
        y1, y2 = self.__size(y)
        j = 0
        for i in range(0, y2):
            y, j = redu2(j, y, c, d, x)
        return y


