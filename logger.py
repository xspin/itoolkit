import time

class Clock:
    def __init__(self, n_steps=None):
        self.tic()
        self.n_steps = n_steps
    def tic(self):
        self.start_time = time.time()
    def toc(self, step=None):
        cost = time.time() - self.start_time
        if step is None or self.n_steps is None: 
            return self._sec2str(cost)
        else:
            step += 1
            return self._sec2str(cost), self._sec2str(cost*(self.n_steps-step)/step)

    def _sec2str(self, sec):
        t = sec
        s, t = t%60, t//60
        m, t = t%60, t//60
        h, d = t%24, t//24
        if d > 0: return "{:.0f}d {:.0f}h {:.0f}m {:.0f}s".format(d,h,m,s)
        if h > 0: return "{:.0f}h {:.0f}m {:.0f}s".format(h,m,s)
        if m > 0: return "{:.0f}m {:.0f}s".format(m,s)
        return "{:.02f}s".format(s)

if __name__ == "__main__":
    timer = Clock(10)
    timer.tic()
    for i in range(10):
        time.sleep(3)
        toc = timer.toc(i)
        print('Elapsed {}  ETA {}'.format(*toc))