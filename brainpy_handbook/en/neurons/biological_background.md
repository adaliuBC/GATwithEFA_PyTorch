## 1.1 Biological backgrounds

As the basic unit of neural systems, neurons maintain mystique to researchers for a long while. In recent centuries, however, along with the development of experimental techniques, researchers have painted a general figure of those little things working ceaselessly in our neural system.

To achieve our final goal of modeling neurons with computational neuroscience methods, we may start with a patch of real neuron membrane.

<center><img src="../../figs/neus/neuron_membrane.png" width="584" height="300">	</center>

<center><b> Fig. 1-1 Neuron membrane diagram | what-when-how.com </b></center>

The figure above is a general diagram of neuron membrane with phospholipid bilayer and ion channels. The membrane divides the ions and fluid into intracellular and extracellular, partially prevent them from exchanging, thus generates **membrane potential**---- the difference in electric potential across the membrane.

An ion in the fluid is subjected to two forces. The force of diffusion is caused by the ion concentration difference across the membrane, while the force of electric field is caused by the electric potential difference. When these two forces reach balance, the total forces on ions are 0, and each type of ion meets an equilibrium potential, while the neuron holds a membrane potential lower than 0.

This membrane potential integrated by all those ion equilibrium potentials is the **resting potential**, and the neuron is, in a so-called **resting state**. If the neuron is not disturbed, it will just come to the balanced resting state, and rest.

However, our neural system receives countless inputs every millisecond, from external inputs to recurrent inputs, from specific stimulus inputs to non-specific background inputs. Receiving all these inputs, neurons generate **action potentials** (or **spikes**) to transfer and process information all across the neural system.

<center><img src="../../figs/neus/action_potential.png" width="533" height="300">	</center>

<center><b> Fig. 1-2 Action Potential | Wikipedia </b></center>

Passing through the ion channels shown in Fig.1-1, ions on both sides of the hydrophobic phospholipid bilayer are exchanged. Due to changes in the environment caused by, for example, an external input, ion channels will switch between their open/close states, therefore allow/prohibit ion exchanges. During the switch, the ion concentrations (mainly Na+ and K+) change, induce a significant change on neuron's membrane potential: the membrane potential will raise to a peak value and then fall back in a short time period. Biologically, when such a series of potential changes happens, we say the neuron generates an **action potential** or a **spike**, or the neuron fires.

An action potential can be mainly divided into three periods: **depolarization**, **repolarization** and **refractory period**. During the depolarization period, Na+ flow into the neuron and K+ flow out of the neuron, however the inflow of Na+ is faster, so the membrane potential raises from a low value $$V_{rest}$$ to a value much higher called $$V_{th}$$, then the outflow of K+ becomes faster than Na+, and the membrane potential is reset to a value lower than resting potential during the repolarization period. After that, because of the relatively lower membrane potential, the neuron is unlikely to generate another spike immediately, until the refractory period passes.

A single action potential is complex enough, but in our neural system, one single neuron can generate several action potentials in less than a second. How, exactly, do the neurons fire? Different kinds of neurons may spike when facing different inputs, and the pattern of their spiking can be classified into several firing patterns, some of which are shown in the following figure.

<center><img src="../../figs/neus/multiple_firing_pattern.png" width="600" height="300">	</center>

<center><b>Figure 1-3 Some firing patterns</b></center>

Those firing patterns, together with the shape of action potentials, are what computational neuroscience wants to model at the cellular level.