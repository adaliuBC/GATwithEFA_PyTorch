# 【附录补充】更多化学突触模型

### NMDA

NMDA的公式如下，
$$ \frac{d s_{j}(t)}{dt} =-\frac{s_{j}(t)}{\tau_{decay}}+a x_{j}(t)(1-s_{j}(t))  $$

$$ \frac{d x_{j}(t)}{dt} =-\frac{x_{j}(t)}{\tau_{rise}}+
\sum_{k} \delta(t-t_{j}^{k})  $$

$$ g_{\infty}(V,[{Mg}^{2+}]_{o}) =(1+{e}^{-\alpha V} \cdot \frac{[{Mg}^{2+}]_{o} } {\beta})^{-1}  $$

$$ g(t) = \bar{g}_{syn} \cdot g_{\infty}  s $$

其中，它的E为0

NMDA实现的代码如下


```python
class NMDA(bp.TwoEndConn):    
    target_backend = 'general'

    @staticmethod
    def derivative(s, x, t, tau_rise, tau_decay, a):
        dxdt = -x / tau_rise
        dsdt = -s / tau_decay + a * x * (1 - s)
        return dsdt, dxdt
    
    def __init__(self, pre, post, conn, delay=0., g_max=0.15, E=0., cc_Mg=1.2,
                    alpha=0.062, beta=3.57, tau=100, a=0.5, tau_rise = 2., **kwargs):
        # parameters
        self.g_max = g_max
        self.E = E
        self.alpha = alpha
        self.beta = beta
        self.cc_Mg = cc_Mg
        self.tau = tau
        self.tau_rise = tau_rise
        self.a = a
        self.delay = delay

        # connections
        self.conn = conn(pre.size, post.size)
        self.conn_mat = conn.requires('conn_mat')
        self.size = bp.ops.shape(self.conn_mat)

        # variables
        self.s = bp.ops.zeros(self.size)
        self.x = bp.ops.zeros(self.size)
        self.g = self.register_constant_delay('g', size=self.size, delay_time=delay)

        self.integral = bp.odeint(f=self.derivative, method='rk4')
        
        super(NMDA, self).__init__(pre=pre, post=post, **kwargs)


    def update(self, _t):
        self.x += bp.ops.unsqueeze(self.pre.spike, 1) * self.conn_mat        
        self.s, self.x = self.integral(self.s, self.x, _t, self.tau_rise, self.tau, self.a)
        
        self.g.push(self.g_max * self.s)
        g_inf = 1 + self.cc_Mg / self.beta * bp.ops.exp(-self.alpha * self.post.V)
        g_inf = 1 / g_inf
        self.post.input -= bp.ops.sum(self.g.pull(), axis=0) * (self.post.V - self.E) * g_inf
```

### GABA_a

GABA_a 可以采用single exponential的形式来实现，代码如下：


```python
class GABAa(bp.TwoEndConn):
    target_backend = 'general'

    @staticmethod
    def derivative(s, t, tau_decay):
        dsdt = - s / tau_decay
        return dsdt

    def __init__(self, pre, post, conn, delay=0.,
                 g_max=0.4, E=-80., tau=6.,
                 **kwargs):
        # parameters
        self.g_max = g_max
        self.E = E
        self.tau = tau
        self.delay = delay

        # connections
        self.conn = conn(pre.size, post.size)
        self.conn_mat = conn.requires('conn_mat')
        self.size = bp.ops.shape(self.conn_mat)

        # data
        self.s = bp.ops.zeros(self.size)
        self.g = self.register_constant_delay('g', size=self.size,
                                              delay_time=delay)

        self.integral = bp.odeint(f=self.derivative, method='euler')
        super(GABAa, self).__init__(pre=pre, post=post, **kwargs)

    def update(self, _t):
        self.s = self.integral(self.s, _t, self.tau)
        self.s += bp.ops.unsqueeze(self.pre.spike, 1) * self.conn_mat
        self.g.push(self.g_max * self.s)
        self.post.input -= bp.ops.sum(self.g.pull(), axis=0) \
                           * (self.post.V - self.E)
```

### GABA_b

GABA_b的公式如下:

$$
\frac{d[R]}{dt} = k_3 [T](1-[R])- k_4 [R]
$$

$$
\frac{d[G]}{dt} = k_1 [R]- k_2 [G]
$$

$$
I_{GABA_{B}} =\overline{g}_{GABA_{B}} (\frac{[G]^{4}} {[G]^{4}+K_{d}}) (V-E_{GABA_{B}})
$$


```python
class GABAb(bp.TwoEndConn):    
    target_backend = 'general'

    @staticmethod
    def derivative(G, R, t, k1, k2, k3, k4, TT):
        dGdt = k1 * R - k2 * G
        dRdt = k3 * TT * (1 - R) - k4 * R
        return dGdt, dRdt

    def __init__(self, pre, post, conn, delay=0.,
                 g_max=0.02, E=-95., k1=0.18, k2=0.034,
                 k3=0.09, k4=0.0012, kd=100.,
                 T=0.5, T_duration=0.3, **kwargs):
        self.g_max = g_max
        self.E = E
        self.k1 = k1
        self.k2 = k2
        self.k3 = k3
        self.k4 = k4
        self.kd = kd
        self.T = T
        self.T_duration = T_duration
        self.delay = delay

        self.conn = conn(pre.size, post.size)
        self.conn_mat = self.conn.requires('conn_mat')
        self.size = bp.ops.shape(self.conn_mat)

        self.R = bp.ops.zeros(self.size)
        self.G = bp.ops.zeros(self.size)
        self.s = bp.ops.zeros(self.size)
        self.g = self.register_constant_delay('g', size=self.size, delay_time=delay)
        self.t_last_pre_spike = bp.ops.ones(self.size) * -1e7

        self.integral = bp.odeint(f=self.derivative, method='euler')
        super(GABAb, self).__init__(pre=pre, post=post, **kwargs)

    def update(self, _t):
        spike = bp.ops.unsqueeze(self.pre.spike, 1) * self.conn_mat
        self.t_last_pre_spike = bp.ops.where(spike, _t, self.t_last_pre_spike)
        TT = ((_t - self.t_last_pre_spike) < self.T_duration) * self.T
        self.G, self.R = self.integral(
            self.G, self.R, _t,
            self.k1, self.k2,
            self.k2, self.k4, TT)
        self.s = self.G ** 4 / (self.G ** 4 + self.kd)
        self.g.push(self.g_max * self.s)
        self.post.input -= bp.ops.sum(self.g.pull(), 0) * (self.post.V - self.E)
```

Let's compare the 4 synapses!


```python
(I_ext, duration) = bp.inputs.constant_current([(0, 1), (35, 15), (0, 300)])

# AMPA
neu = bm.neurons.LIF(1)
ampa = AMPA(pre=neu, post=neu, conn=bp.connect.All2All(), monitors=['s'])
net_ampa = bp.Network(neu, ampa)
net_ampa.run(duration, inputs=(neu, "input", I_ext))

# NMDA
neu = bm.neurons.LIF(1)
nmda = NMDA(pre=neu, post=neu, conn=bp.connect.All2All(), monitors=['s'])
net_nmda = bp.Network(neu, nmda)
net_nmda.run(duration, inputs=(neu, "input", I_ext))

# GABA_a
neu = bm.neurons.LIF(1)
gabaa = GABAa(pre=neu, post=neu, conn=bp.connect.All2All(), monitors=['s'])
net_gabaa = bp.Network(neu, gabaa)
net_gabaa.run(duration, inputs=(neu, "input", I_ext))

# GABA_b
neu = bm.neurons.LIF(1)
gabab = GABAb(pre=neu, post=neu, conn=bp.connect.All2All(), monitors=['s'])
net_gabab = bp.Network(neu, gabab)
net_gabab.run(duration, inputs=(neu, "input", I_ext))


# visualization
ts = net_nmda.ts

plt.plot(ts, nmda.mon.s[:, 0]* 0.15 * (-65 - 0) * -50, label='NMDA')
plt.plot(ts, ampa.mon.s[:, 0]* 0.45 * (-65 - 0) * -50, label='AMPA')
plt.plot(ts, gabab.mon.s[:, 0]* (-65 + 95) * -200 * 1e+7, label='GABA_b')
plt.plot(ts, gabaa.mon.s[:, 0] * 0.4 * (-65 + 80) * -50, label='GABA_a')
plt.ylabel('-I')
plt.xlabel('Time (ms)')
plt.legend()
plt.show()
```


![png](/home/nipcora/Documents/Brain-Models-book/figs/out/output_103_0.png)

