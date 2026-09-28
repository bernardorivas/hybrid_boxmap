# Proofs and computed checks for the four examples

Date: September 27, 2026  
Status: Reference for readers of the code. Section "Examples" of the paper illustrates the computations without stating results about the examples. This file keeps the statements and arguments of an earlier draft of that section and compares the recorded runs with the invariant sets and indices that these arguments give. The Morse graphs and labels come from sampled multivalued maps, not certified outer approximations, so the comparisons are consistency checks, not proofs.  
Scope: `lem:window-isolation`, `prop:ball-window`, `prop:wheel-isolation`, and `lem:impact-lienard`; the invariant sets and indices of the bouncing ball, the rimless wheel, the spiking neuron, and the impacting oscillator; the four runs in `figures/paper/` listed under "Runs".  
Source: an earlier draft of Section 6, "Examples", of the paper (September 27, 2026). Computed facts are taken from the JSON run records, recorded at code `a47c88d`. The Conley index of the Morse set of $U_Z$ in the oscillator run was computed afterwards, in commit `02270ea`.

Citations are given by their keys in the bibliography of the paper. The labels `lem:window-isolation`, `prop:ball-window`, `prop:wheel-isolation`, and `lem:impact-lienard` are those of the earlier draft. Other labels are those of the paper.

## Setting and notation

A hybrid system $\mathcal H=(X,\varphi,G,r)$ has a compact metric phase space $X$, a closed guard $G\subseteq X$, a local semiflow $\varphi$ on $X\setminus G$, and a continuous reset $r:G\to X$ (`def:hybrid-system`), with impact time $\sigma$ and impact map $p$. The suspension $\Sigma X$ is the quotient of $X\sqcup(G\times[0,1])$ by $(g,0)\sim g$ and $(g,1)\sim r(g)$, with quotient map $\pi$ and inclusion $\tilde\iota:X\hookrightarrow\Sigma X$. For $x\in G$, the set $\pi(\lbrace x\rbrace\times[0,1])$ is the handle over $x$. The hybrid suspension semiflow $\Phi$ follows $\varphi$ on $\tilde\iota(X)$ and spends one unit of time on each handle, and $f_\tau:=\Phi(\tau,\cdot)$ is its $\tau$-map. The space $\Sigma X$ is compact and metrizable, and under the trapping guard condition (TGC) the semiflow $\Phi$ is continuous. For $A\subseteq X$, the set $\Sigma(A)$ consists of $\tilde\iota(A)$ and the handles over $A\cap G$. For $0<b<a<1$ and $U\subseteq X$,

$$
W_{a,b}(U):=\tilde\iota(U)\cup\pi\left((U\cap G)\times[0,a]\right)\cup\pi\left(r^{-1}(U)\times[b,1]\right).
$$

For $U\subseteq\Sigma X$ and $I\subseteq\mathbb R_{\geq0}$, let $\Phi(I,U):=\lbrace\Phi(t,x)\mid t\in I,\ x\in U\rbrace$ and

$$
\omega(U,\Phi):=\bigcap_{T\geq0}\operatorname{cl}\left(\Phi([T,\infty),U)\right).
$$

The set $U$ is an attracting neighborhood if $\omega(U,\Phi)\subseteq\operatorname{int}U$, and these sets form $\mathsf{ANbhd}(\Sigma X,\Phi)$. For compact $B$, the set $\operatorname{Inv}(B,f)$ consists of the points lying on full $f$-orbits in $B$, and $B$ is isolating for $f$ if $\operatorname{Inv}(B,f)\subseteq\operatorname{int}B$. The hybrid Conley index of [`Rivas:Kalies`] is written $\Sigma\mathrm{CH}_{*}$.

### The computation

Given a compact set $R\subseteq X$ with $r(G\cap R)\subseteq R$, every computation takes place on

$$
\widetilde D:=\Sigma(R)=\tilde\iota(R)\cup\pi\left((G\cap R)\times[0,1]\right).
$$

For the bouncing ball, the rimless wheel, and the impacting oscillator, $R$ is a rectangle, and for the spiking neuron $R=X$. The suspension grid $\Xi_n$ on $\widetilde D$ has the base grid $\mathcal X_n$, whose cells are the nonempty sets $\operatorname{cl}\operatorname{int}(B'\cap R)$ for the cells $B'$ of the subdivision of a rectangle $B\supseteq R$ into $2^{n+4}\times2^{n+4}$ congruent rectangles ($2^{n+3}\times2^{n+3}$ for the neuron), and the phase grid of $2^{n+2}$ intervals of width $a_n=2^{-n-2}$. The multivalued map $\mathcal F:\Xi_n\rightrightarrows\Xi_n$ is sampled at the corners of the pieces, with sides halved at most $14$ times where the image points of two corners lie in elements that do not meet, and it sends $\xi$ to the elements that meet an element containing the image point of a sample of $\xi$. Image points outside $\widetilde D$ are discarded.

The Morse sets $M$ of $\mathcal F$ are its recurrent strongly connected components, ordered by reachability, and the Morse graph $\operatorname{MG}(\mathcal F)$ is drawn as its transitive reduction. The label of $M$ is the shift equivalence class, over $\mathbb F=\mathbb F_5$, of the index map that $\mathcal F$ induces on

$$
H_{*}\left(|M\cup\mathcal F(M)|_{\Sigma},\ |\mathcal F(M)\setminus M|_{\Sigma};\ \mathbb F\right).
$$

It lists the invariant factors of the restriction of the index map to its eventual image in each degree, from $0$ to $3$ ($5$ for the neuron). Here $x-1$ denotes the identity on $\mathbb F$ and $0$ the zero module. When the relative homology vanishes, the label is trivial without an index map (`label_source: "zero relative homology"` in the JSON). Since $\mathcal F$ is not known to be an outer approximation, the labels are not covered by `prop:grid-conley-index`. For an outer approximation, that proposition shows that $(f_\tau)_{*}$ is the identity on its eventual image, so every invariant factor of the Conley index of $f_\tau$ on the isolated invariant sets of that proposition is $x-1$. The base cells of $M$ are the cells of $|d_n^{-1}(M)|\subseteq R$, which form the regularized preimage of $|M|_\Sigma$ (`prop:finite-grid-preimage`).

## Isolation of $\widetilde D$

Discarding image points outside $\widetilde D$ loses no invariant dynamics when $\widetilde D$ is isolating for $f_\tau$, that is, when $\operatorname{Inv}(\widetilde D,f_\tau)\subseteq\operatorname{int}_{\Sigma X}\widetilde D$. If $\mathcal F$ were an outer approximation of $f_\tau$ on $\widetilde D$, its Morse graph would then describe a Morse representation of $\operatorname{Inv}(\widetilde D,f_\tau)$ [`Arai:Kalies:Kokubu:Mischaikow:Oka:Pilarczyk`]. For the bouncing ball, the rimless wheel, and the impacting oscillator, $\widetilde D$ is not forward invariant, so it is not a trapping region for $f_\tau$ when $\tau$ is small. Isolation nevertheless holds whenever the orbits starting in $\widetilde D$ enter and remain in its interior after a uniform time.

### Lemma (`lem:window-isolation`)

For a compact set $\widetilde D\subseteq\Sigma X$, the following statements are equivalent.

- (i) $\widetilde D\in\mathsf{ANbhd}(\Sigma X,\Phi)$.
- (ii) There exists $T\geq0$ such that $\Phi([T,\infty),\widetilde D)\subseteq\operatorname{int}_{\Sigma X}\widetilde D$.

In this case, $\operatorname{Inv}(\widetilde D,f_\tau)=\omega(\widetilde D,\Phi)\subseteq\operatorname{int}_{\Sigma X}\widetilde D$ for every $\tau>0$.

Proof. If (i) holds, the compact sets $\operatorname{cl}_{\Sigma X}\Phi([t,\infty),\widetilde D)$ decrease in $t$, and their intersection $\omega(\widetilde D,\Phi)$ lies in the open set $\operatorname{int}_{\Sigma X}\widetilde D$. Hence one of them lies in $\operatorname{int}_{\Sigma X}\widetilde D$, which proves (ii). Conversely, if (ii) holds, then $\omega(\widetilde D,\Phi)\subseteq\operatorname{cl}_{\Sigma X}\Phi([T,\infty),\widetilde D)\subseteq\widetilde D$. Since $\omega(\widetilde D,\Phi)$ is invariant,

$$
\omega(\widetilde D,\Phi)=\Phi\left(T,\omega(\widetilde D,\Phi)\right)\subseteq\Phi(T,\widetilde D)\subseteq\operatorname{int}_{\Sigma X}\widetilde D,
$$

which proves (i).

Now let $x\in\operatorname{Inv}(\widetilde D,f_\tau)$. For every $k\geq0$ there is $x_k\in\widetilde D$ with $f_\tau^k(x_k)=x$, so $x\in\Phi(k\tau,\widetilde D)$ for every $k$, and therefore $x\in\omega(\widetilde D,\Phi)$. Conversely, $\omega(\widetilde D,\Phi)\subseteq\widetilde D$ is invariant under $\Phi$, so $f_\tau$ maps it onto itself and each of its points lies on a full $f_\tau$-orbit in $\widetilde D$. $\square$

Consequently, when $\widetilde D$ satisfies (ii), the set $\operatorname{Inv}(\widetilde D,f_\tau)=\omega(\widetilde D,\Phi)$ does not depend on $\tau$, even for $\tau<T$, where $f_\tau(\widetilde D)$ can leave $\widetilde D$. The choice of $\tau$ is limited by the resolution instead. Where $f_\tau$ moves points by less than a cell, the enlargement by one layer of elements lets neighboring cells map to each other, and they form large strongly connected components. So $\tau$ is chosen small, but large enough that the $\tau$-map is not close to the identity at the resolution of the grid. The bouncing ball satisfies (ii) by `prop:ball-window`, and numerically so does the impacting oscillator, while for the spiking neuron $\widetilde D=\Sigma X$. For the rimless wheel, $\widetilde D$ contains a saddle whose unstable manifold leaves it, so isolation is verified directly in `prop:wheel-isolation`.

The lemma uses the standing hypotheses of the manuscript: $X$ is compact and $\Phi$ is a continuous semiflow, so that the sets $\operatorname{cl}_{\Sigma X}\Phi([t,\infty),\widetilde D)$ are compact and $\omega(\widetilde D,\Phi)$ is invariant. The rimless wheel does not satisfy them, and for the impacting oscillator compactness of $X$ holds only if the numerical inclusion of that section holds.

## Runs

| Example | Stem in `figures/paper/` | $\tau$ | Grid | Base cells per axis (cell size) | Phase intervals | Morse sets |
|---|---|---|---|---|---|---|
| bouncing ball | `paper-bouncing-ball-tau050-level6-base1024-corners-gap-refined` | 0.5 | $\Xi_6$ | 1024 (0.001953 x 0.009766) | 256 | 1 |
| rimless wheel | `paper-rimless-wheel-tau050-level7-base2048-corners-gap-refined` | 0.5 | $\Xi_7$ | 2048 (0.0003906 x 0.0007324) | 512 | 17 |
| spiking neuron | `paper-spiking-neuron-tau500-level7-base1024-corners-gap-refined` | 5 | $\Xi_7$ | 1024 of $B$ (0.3125 x 1.25) | 512 | 1 |
| impacting oscillator, $\beta=0.76$ | `paper-impact-vdp-duffing-beta076-tau050-level7-base2048-corners-gap-refined` | 0.5 | $\Xi_7$ | 2048 (0.001343 x 0.002100) | 512 | 22 |

Each stem has the run record `<stem>.json` and the following figures, as PDF. `demo/replot_paper.py` also writes PNG copies.

- `<stem>.pdf`: every Morse set, with the base cells, the zooms, and the Morse graph.
- `<stem>-nontrivial.pdf`: the same, without the Morse sets whose computed label is trivial, ordered by reachability in $\operatorname{MG}(\mathcal F)$ along paths that may pass through the hidden Morse sets.
- Panels of `<stem>.pdf`: `<stem>-base.pdf`, `<stem>-zoom-A.pdf`, `<stem>-zoom-B.pdf`, ..., and `<stem>-graph.pdf`, the Morse graph of all Morse sets, in which the sets whose computed label is trivial are gray.
- Panels of `<stem>-nontrivial.pdf`: `<stem>-nontrivial-base.pdf`, `<stem>-nontrivial-zoom-A.pdf`, ..., and `<stem>-nontrivial-graph.pdf`.

The figures have no panel of handle pieces. The node numbers $M(i)$ below are the indices of `morse_graph.nodes` in the JSON, and an edge $M(i)\to M(j)$ is the pair `[i, j]` of `morse_graph.edges`, meaning that $M(i)$ reaches $M(j)$. In every run, the grid check passed, no sample failed, no side remained unresolved after halving, no padded image is disconnected, and none of the endpoint probes of the manuscript (a $5\times5$ array of points in each of $150$ random base cells and $7$ phases over one point of each of $20$ random guard intervals) was missed.

## Bouncing ball

The bouncing ball of [`Rivas:Kalies`] has height $h$, velocity $v$, flow $(\dot h,\dot v)=(v,-g)$, guard $G=\lbrace(0,v)\mid v\leq0\rbrace$, and reset $r(0,v)=(0,-cv)$, where $g=9.81$ and $c=0.8$. The energy $E(h,v):=gh+v^2/2$ is constant along the flow and is multiplied by $c^2$ at each reset. Here $R:=[0,2]\times[-5,5]$, and the phase space

$$
X:=\lbrace(h,v)\mid h\geq0,\ E(h,v)\leq E_R\rbrace,\quad E_R:=2g+25/2,
$$

is the smallest set of this form that contains $R$. Note that $X$ is compact and forward invariant, since $E$ does not increase along orbits. The impact time $\sigma(h,v)=\left(v+\sqrt{v^2+2gh}\right)/g$ and the impact map $p(h,v)=\left(0,-\sqrt{v^2+2gh}\right)$ are continuous on $X$, so $\mathcal H$ satisfies TGC by `prop:trapping-guard-characterizations`. Since $r(0,0)=(0,0)$, the Zeno point at the origin corresponds to the periodic orbit $\widetilde Z:=\Sigma(\lbrace(0,0)\rbrace)$ of $\Phi$, which has period $1$. Recall that $\Sigma\mathrm{CH}_{*}(\lbrace(0,0)\rbrace)\cong H_{*}(S^1)$ [`Rivas:Kalies`].

### Proposition (`prop:ball-window`)

The set $\widetilde D=\Sigma(R)$ satisfies $\Phi([T,\infty),\widetilde D)\subseteq\operatorname{int}_{\Sigma X}\widetilde D$ for $T=8.52$, and $\omega(\widetilde D,\Phi)=\widetilde Z$. In particular, $\operatorname{Inv}(\widetilde D,f_\tau)=\widetilde Z$ for every $\tau>0$.

Proof. Our strategy is as follows. We specify an energy $E_1$ below which every state lies in $\operatorname{int}_{\Sigma X}\widetilde D$, and a time by which every orbit starting in $\widetilde D$ has fallen below $E_1$. Define $E_\Sigma:\Sigma X\to\mathbb R_{\geq0}$ by $E_\Sigma(\tilde\iota(x)):=E(x)$ for $x\in X$ and $E_\Sigma(\pi(x,s)):=E(x)$ for $x\in G$ and $0<s<1$. Then $E_\Sigma$ is constant along the flow and on each handle, and it is multiplied by $c^2$ at each reset, so it is nonincreasing along $\Phi$. Moreover, $E_\Sigma$ is lower semicontinuous. Indeed, it is continuous except at the points $\pi(x,1)=\tilde\iota(r(x))$ with $E(x)>0$, where the nearby handle points carry values close to $E(x)>c^2E(x)=E_\Sigma(\pi(x,1))$. Hence the sets $\lbrace z\in\Sigma X\mid E_\Sigma(z)\leq e\rbrace$ with $e\geq0$ are closed.

Let $E_1:=\min\lbrace 2g,\,25c^2/2\rbrace=8$ and $U:=\lbrace x\in X\mid E(x)<E_1\rbrace$. The set $U$ is open in $X$ and forward invariant, so `lem:Wab`(iii) yields $\Sigma(U)\subseteq\operatorname{int}_{\Sigma X}W_{a,b}(U)$ for $0<b<a<1$. Moreover, $U\subseteq[0,E_1/g)\times(-4,4)\subseteq R$ and $r^{-1}(U)\subseteq\lbrace0\rbrace\times(-5,0]\subseteq G\cap R$, since $c^2v^2/2<E_1$ implies $|v|<4/c=5$. Hence $W_{a,b}(U)\subseteq\widetilde D$, and

$$
\lbrace z\in\Sigma X\mid E_\Sigma(z)<E_1\rbrace=\Sigma(U)\subseteq\operatorname{int}_{\Sigma X}\widetilde D.
$$

Note that $W_{a,b}(U)$ also contains handle points over guard points $(0,v)$ with $E(0,v)\geq E_1$, which lie in $\widetilde D$ since $|v|<5$.

Every orbit starting in $\widetilde D$ completes $k$ resets by the time

$$
T_k:=\frac{5+\sqrt{2E_R}}{g}+k+\frac{2\sqrt{2E_R}}{g}\sum_{j=1}^{k-1}c^j,
$$

since a state on a handle reaches the end of its handle within time $1$, the first impact from $(h,v)\in R$ occurs within time $\left(v+\sqrt{2E(h,v)}\right)/g$, each handle takes one unit of time, and the flight after the $j$th reset lasts at most $2c^j\sqrt{2E_R}/g$. After $k$ resets, $E_\Sigma\leq c^{2k}E_R$, so

$$
\Phi([T_k,\infty),\widetilde D)\subseteq\lbrace z\in\Sigma X\mid E_\Sigma(z)\leq c^{2k}E_R\rbrace\quad\forall k\geq1.
$$

Since $c^8E_R<E_1$, it follows that $\Phi([T_4,\infty),\widetilde D)\subseteq\operatorname{int}_{\Sigma X}\widetilde D$, where $T_4<8.52$. Since the sets on the right-hand side are closed, $\omega(\widetilde D,\Phi)$ lies in their intersection, which is $\lbrace z\in\Sigma X\mid E_\Sigma(z)=0\rbrace=\widetilde Z$. Conversely, $\widetilde Z\subseteq\widetilde D$ is invariant, so $\widetilde Z\subseteq\Phi(t,\widetilde D)$ for every $t\geq0$ and hence $\widetilde Z\subseteq\omega(\widetilde D,\Phi)$. The last statement follows from `lem:window-isolation`. $\square$

For reference, $E_R=32.12$, $c^8E_R=5.389$, $c^6E_R=8.420>E_1$ (so $k=4$ is the first $k$ that works), and $T_4=8.5164$.

Note that $f_\tau(\widetilde D)\not\subseteq\widetilde D$ for $\tau=0.5$. The state $(2,-5)$ reaches the guard at time $0.31$ with $v=-\sqrt{2E_R}\approx-8.01<-5$, so at time $0.5$ it lies on the handle over a guard point outside $R$.

### Comparison with the run

By `prop:ball-window`, $\operatorname{Inv}(\widetilde D,f_{0.5})=\widetilde Z$, a single periodic orbit with index $H_{*}(S^1)$.

| | Analysis | Run ($\tau=0.5$, $\Xi_6$, 1024 cells) |
|---|---|---|
| Morse graph | one Morse set, $\widetilde Z$ | one Morse set $M(0)$, no edge |
| location | the origin and the handle over it | 105 base cells in $[0,0.003906]\times[-0.2148,0.3027]$, which contains the origin, and 13,165 handle pieces meeting all 256 phase intervals; one component in $\Sigma X$ |
| index | $\Sigma\mathrm{CH}_{*}(\lbrace(0,0)\rbrace)\cong H_{*}(S^1)$ | label $(x-1,x-1,0,0)$ |

The pair of $M(0)$ has $\mathcal F(M)\setminus M=\emptyset$ and homology of dimensions $(1,2,0,0)$. In degree $1$ the index map has a one-dimensional eventual image, on which it is the identity, so the label has the form of $H_{*}(S^1)$. The manuscript states that $M(0)$ contains every element of $\Xi_6$ that meets $\widetilde Z$. The JSON has no `reference_set_identification` for this run, and this containment rests on the separate check described in `PAPER_GRID.md`, "Recommended configurations". Both figure variants are the same, since the only Morse set has a nontrivial label.

## Rimless wheel

The rimless wheel model of [`Shia:Vasudevan:Bajcsy:Tedrake`], a planar wheel with spokes rolling down an inclined plane [`Coleman`], is used with the parameters of [`Rivas:Kalies`]. Its state consists of the angle $\theta$ between the vertical and the spoke in contact with the ground and the velocity $\dot\theta$. The flow is $\ddot\theta=\sin\theta$, the guard is $G=\lbrace(\alpha+\gamma,\dot\theta)\mid\dot\theta\geq0\rbrace$, and the reset is $r(\theta,\dot\theta)=(2\gamma-\theta,\cos(2\alpha)\dot\theta)$, where the slope is $\gamma=0.2$ and the angle between two spokes is $2\alpha=0.8$. The energy $E(\theta,\dot\theta):=\dot\theta^2/2+\cos\theta$ is constant along the flow, and a reset sends pre-impact energy $e$ to the post-reset energy

$$
P(e):=\cos^2(0.8)\,e+\cos(0.2)-\cos^2(0.8)\cos(0.6).
$$

The map $P$ is affine with slope $\cos^2(0.8)<1$, so it has a unique fixed point $E_*\approx1.1260$, which determines the walking gait. The gait corresponds to a periodic orbit $\widetilde\Gamma$ of $\Phi$, which meets the guard at $\dot\theta\approx0.7755$ and resets to $(-0.2,0.5403)$.

Here $R:=[-0.2,0.6]\times[-0.5,1]$, which contains the saddle at the origin. The right branch of its unstable manifold has energy $1$, meets the guard at $\dot\theta\approx0.5910$, resets to $(-0.2,0.4118)$, and converges to the gait. It corresponds to an orbit $\widetilde W$ of $\Phi$. The left branch rolls backward past $\theta=-0.2$ and never returns.

Let $X$ be the closure of the set of states reached from $R$. An orbit from $R$ reaches the line $\theta=0.6$ only through $G$, where it is reset, so $X\subseteq\lbrace\theta\leq0.6\rbrace$. As a closure, $X$ is closed, but it is not compact, since the states of $R$ with $\dot\theta<0$ and $E>1$ roll backward indefinitely. The wheel therefore does not satisfy `def:hybrid-system`, and the results of the manuscript's Sections 3 to 5 do not apply to it, so the computation is treated as local. Note that $\mathcal H$ satisfies TGC on $X$. Indeed, the flow crosses $G$ transversally except at the grazing point $(0.6,0)$, and near this point every state of $X$ has $\theta\leq0.6$ and $\ddot\theta=\sin\theta>0$, so it reaches $G$ in a time that tends to $0$ as the state approaches $(0.6,0)$. On the plane, TGC fails at $(0.6,0)$, since the states $(0.6+\delta,0)$ with $\delta>0$ oscillate about $\theta=\pi$ and never reach $G$. The following proposition concerns only the orbits of $\Phi$ and the topology of $\Sigma X$, so it uses neither compactness of $X$ nor continuity of $\Phi$.

### Proposition (`prop:wheel-isolation`)

For $0<\tau\leq5$, the set $\widetilde D=\Sigma(R)$ satisfies

$$
\operatorname{Inv}(\widetilde D,f_\tau)=\lbrace\tilde\iota(0,0)\rbrace\cup\widetilde W\cup\widetilde\Gamma\subseteq\operatorname{int}_{\Sigma X}\widetilde D.
$$

Proof. Our strategy is as follows. We first show that the three orbits lie in $\operatorname{int}_{\Sigma X}\widetilde D$, and then follow a full $f_\tau$-orbit in $\widetilde D$ backward in time to show that it lies on one of them.

Along $\widetilde\Gamma$ we have $0.502\leq\dot\theta\leq0.776$, and along $\widetilde W$ we have $0\leq\dot\theta\leq0.776$. Hence both orbits avoid the horizontal edges of $R$ and meet the guard at $\dot\theta<1$, in the interior of $G\cap R$ relative to $G$. Every state of $X$ near these guard points lies in $R$, since $X\subseteq\lbrace\theta\leq0.6\rbrace$. The orbits reach the edge $\theta=-0.2$ only at their reset points, where $0.411\leq\dot\theta\leq0.541$, and every state of $X$ near these points also lies in $R$. Indeed, a state of $X$ with $\theta<-0.2$ either satisfies $\dot\theta<0$ and never returns, or oscillates about $\theta=-\pi$ with $E<1$. Near $\theta=-0.2$ such an oscillation satisfies $|\dot\theta|<0.2$, since $\sqrt{2(1-\cos(0.2))}<0.2$. Hence the three orbits lie in $\operatorname{int}_{\Sigma X}\widetilde D$.

Conversely, let $z\in\operatorname{Inv}(\widetilde D,f_\tau)$ lie on the full $f_\tau$-orbit $(z_k)_{k\in\mathbb Z}$ in $\widetilde D$. Concatenating the segments $\Phi([0,\tau],z_k)$ yields a full orbit $\zeta$ of $\Phi$ through $z$ that lies in $\widetilde D$ at the times $k\tau$ and in $\Sigma X$ at all times. Note that $E\leq3/2$ on $\widetilde D$, where a handle point carries its pre-reset energy, and that every pre-impact energy is at least $\cos(0.6)$. Suppose first that $\zeta$ passes infinitely many resets in negative time. Its pre-impact energies are then $P^{-j}(e)$, $j\geq0$, for a single value $e$. Since $P^{-1}$ expands about $E_*$, these values fall below $\cos(0.6)$ if $e<E_*$, which is impossible, and they exceed $3/2$ if $e>E_*$, after which $\zeta$ lies outside $\widetilde D$ at all earlier times. Hence $e=E_*$, and $\zeta$ is $\widetilde\Gamma$.

Suppose instead that $\zeta$ follows the flow for all sufficiently negative times. We work in the strip $[-0.2,0.6]\times\mathbb R$ rather than in $R$, since a backward flow orbit may leave $R$ through a horizontal edge and remain in the strip. The strip contains no compact invariant set of the flow other than the saddle. Indeed, if $K$ is such a set and $|\theta|$ attains its maximum on $K$ at a point with $\theta\neq0$, then $\dot\theta=0$ there, and $\ddot\theta=\sin\theta$ increases $|\theta|$ in both time directions, which is impossible. Hence $K\subseteq\lbrace\theta=0\rbrace$, and invariance forces $\dot\theta=0$ on $K$. Since $E$ is constant along flow orbits, a backward flow orbit that remains in the strip is bounded, and its limit set as $t\to-\infty$ is a compact invariant subset of the strip, so the orbit converges to the saddle. Otherwise the orbit reaches $\lbrace\theta>0.6\rbrace$ or $\lbrace\theta<-0.2\rbrace$. The first of these sets is disjoint from $X$. In the second, the orbit either remains there at all earlier times, when $E\geq1$, or oscillates about $\theta=-\pi$ with $E<1$. Since such an oscillation satisfies $|\dot\theta|<2|\sin(\theta/2)|$, each of its visits to $\lbrace\theta<-0.2\rbrace$ lasts more than

$$
2\int_{0.2}^{\pi}\frac{d\theta}{2\sin(\theta/2)}=-2\ln\tan(0.05)>5.98\geq\tau,
$$

so $\zeta$ lies outside $\widetilde D$ at some time $k\tau$. Therefore $\zeta$ lies in the unstable manifold of the saddle. Since the left branch leaves $\widetilde D$ permanently in forward time, $\zeta$ is the saddle or $\widetilde W$. $\square$

For reference, $E_*=1.126018$, and the constants of the proof evaluate to $\sqrt{2(E_*-\cos0.6)}=0.77548$, $\sqrt{2(E_*-1)}=0.50203$, $\cos(0.8)\cdot0.77548=0.54028$, $\sqrt{2(1-\cos0.6)}=0.59104$, $\cos(0.8)\cdot0.59104=0.41178$, $\sqrt{2(1-\cos0.2)}=0.19967$, and $-2\ln\tan(0.05)=5.9898$.

The indices to compare with are those of the gait, $H_{*}(S^1)$, and of a hyperbolic saddle with one unstable direction, $\mathbb F$ in degree $1$ [`Rivas:Kalies`]. Both comparisons are local. Near $\widetilde\Gamma$ the space $\Sigma X$ agrees with the suspension of the compact phase space $[-0.2,0.6]\times[-1,2]$ of [`Rivas:Kalies`], since every state of $X$ near the reset points of $\widetilde\Gamma$ lies in $R$, and near $\tilde\iota(0,0)$ it is a disk on which $\Phi$ is the flow of $\ddot\theta=\sin\theta$.

### Comparison with the run

By `prop:wheel-isolation`, $\operatorname{Inv}(\widetilde D,f_{0.5})$ consists of the saddle, the gait, and the connecting orbit $\widetilde W$ from the saddle to the gait.

| Set | Index | Morse set | Cells | Label | Homology of the pair |
|---|---|---|---|---|---|
| gait $\widetilde\Gamma$ ($0.502\leq\dot\theta\leq0.776$) | $H_{*}(S^1)$ | $M(0)$ | 54,803 base cells in $[-0.2,0.6]\times[0.4917,0.7847]$; 25,156 handle pieces meeting all 512 phase intervals | $(x-1,x-1,0,0)$ | $(1,1,0,0)$ |
| saddle $\tilde\iota(0,0)$ | $\mathbb F$ in degree 1 | $M(9)$ | 72 base cells in $[-0.003125,0.003125]\times[-0.003418,0.003174]$; no handle piece | $(0,x-1,0,0)$ | $(0,1,0,0)$ |

The order agrees with $\widetilde W$. The figure `<stem>-nontrivial.pdf` shows exactly $M(9)$ and $M(0)$ with the edge $M(9)\to M(0)$. In the full Morse graph, $M(9)$ reaches $M(0)$ along $M(9)\to M(7)\to M(4)\to M(0)$ and $M(9)\to M(8)\to M(6)\to M(3)\to M(0)$.

The other 15 Morse sets, $M(1)$ to $M(8)$ and $M(10)$ to $M(16)$, are single base cells without handle pieces in the bounding box of $M(9)$, within $0.0037$ of the saddle, and none of them contains the saddle. Their pairs have zero relative homology, so their labels are $(0,0,0,0)$, and the figure `<stem>-nontrivial.pdf` hides them. For each $\tau$, $f_\tau$ moves the points close enough to the saddle by less than a cell, so such Morse sets are expected there. Write a point near the saddle as $u\,e_u+s\,e_s$ with the unstable and stable eigenvectors $e_u=(1,1)$ and $e_s=(1,-1)$. The cell centers fall into four groups.

- $u>|s|$, along the right branch of the unstable manifold: $M(3)$, $M(4)$, $M(6)$, $M(7)$, $M(8)$, on the paths from $M(9)$ to $M(0)$ above.
- $-u>|s|$, along the left branch of the unstable manifold: $M(1)$, $M(2)$, $M(5)$, with $M(9)\to M(5)\to M(2)\to M(1)$. The set $M(1)$ reaches no other Morse set, since the images that leave $\widetilde D$ are discarded, and $M(5)$ and $M(2)$ do not reach $M(0)$.
- $-s>2|u|$ and $s>2|u|$, along the two branches of the stable manifold: $M(10)$, $M(12)$, $M(14)$, with $M(14)\to M(12)\to M(10)\to M(9)$, and $M(11)$, $M(13)$, $M(15)$, $M(16)$, with $M(16)\to M(15)\to M(13)\to M(11)\to M(9)$.

No Morse set other than $M(0)$ and $M(9)$ has a nontrivial label, and none lacks a label. The JSON has no `reference_set_identification` for this run. The identification of $M(0)$ and $M(9)$ rests on the extents above and on the separate check described in `PAPER_GRID.md`.

## Spiking neuron

The spiking neuron of [`Rivas:Kalies`] is a quadratic integrate-and-fire model based on [`Izhikevich`], with membrane potential $v$, recovery variable $u$, flow

$$
\begin{aligned}
100\,\dot v&=0.7(v+60)(v+40)-u+70,\\
\dot u&=0.03\left(-2(v+60)-u\right),
\end{aligned}
$$

guard $G=\lbrace(35,u)\mid-300\leq u\leq160\rbrace$, and reset $r(35,u)=(-50,u+100)$. Recall that the L-shaped set

$$
X:=([-80,-40]\times[-300,600])\cup([-40,35]\times[-300,160])
$$

is compact and forward invariant, that $r(G)\subseteq\operatorname{int}X$, that $\mathcal H$ has a periodic orbit, and that the maximal invariant set $A$ of $\mathcal H$ in $X$ has index $\Sigma\mathrm{CH}_{*}(A)\cong H_{*}(S^1)$ [`Rivas:Kalies`]. Note that $\mathcal H$ satisfies TGC. Indeed, on $G$ we have $100\,\dot v=5057.5-u>0$, so the flow crosses $G$ transversally, and $\sigma$ and $p$ are continuous near $G$ (`prop:trapping-guard-characterizations`).

Here $R=X$, so $\widetilde D=\Sigma X$ and no image point is discarded (the JSON records $0$ discarded endpoints). The rectangle $B$ is $[-120,200]\times[-400,880]$. For $n\geq3$, the set $X$ is a union of cells of the subdivision of $B$, so the elements of $\mathcal X_n$ are cells of this subdivision. The cells of $\mathcal X_7$ have size $0.3125\times1.25$. Consequently, the lines $v=-80,-50,-40,35$ and $u=-300,160,600$ are unions of cell faces, and the reset maps the cell faces on $v=35$ onto cell faces on $v=-50$.

The value $\tau=5$ is the smallest of the tested values $\tau=1,2,5,10$ for which $f_\tau$ moves every point of the periodic orbit by more than one cell. The displacement is measured in the maximum norm, with each coordinate in units of the cell side along its axis. At $\tau=5$ it is at least $1.33$, while at $\tau=1$ it is below $1$ for $74\%$ of the period (numbers from the manuscript).

### Comparison with the run

| | Analysis [`Rivas:Kalies`] | Run ($\tau=5$, $\Xi_7$) |
|---|---|---|
| Morse graph | $A$, index $H_{*}(S^1)$ | one Morse set $M(0)$, no edge |
| location | the periodic orbit meets the guard at $u\approx-36.92$ and lies in $[-56.11,35]\times[-36.92,63.08]$ | 6,549 base cells in $[-58.75,35]\times[-48.75,76.25]$ and 10,224 handle pieces meeting all 512 phase intervals; one component in $\Sigma X$ |
| index | $\Sigma\mathrm{CH}_{*}(A)\cong H_{*}(S^1)$ | label $(x-1,x-1,0,0,0,0)$ |

The pair of $M(0)$ has $\mathcal F(M)\setminus M=\emptyset$ and homology of dimensions $(1,1,0,0,0,0)$. The extent of the periodic orbit is from the manuscript. Both figure variants are the same, and there is no zoom.

## Impacting oscillator

This example has several attractors, which are located numerically. A particle moves in the double-well potential $V(x):=x^4/4-x^2/2$ with the damping coefficient $\varepsilon(x^2-\beta)$, which is negative for $|x|<\sqrt\beta$ and positive for $|x|>\sqrt\beta$. The particle hits a rigid stop at $x=w$ inside the right well and bounces back with coefficient of restitution $c$ [`diBernardo:Budd:Champneys:Kowalczyk`]. With velocity $v$ and $x\leq w$, the flow is

$$
\dot x=v,\quad\dot v=x-x^3+\varepsilon(\beta-x^2)v,
$$

the guard is $G=\lbrace(w,v)\mid v\geq0\rbrace$, and the reset is $r(w,v)=(w,-cv)$. The parameters are $\varepsilon=1$, $\beta=0.76$, $w=0.8$, and $c=0.7$. Along the flow, the energy $v^2/2+V(x)$ changes at the rate $\varepsilon(\beta-x^2)v^2$. Since $w<\sqrt\beta$, the flow supplies energy on the whole accessible part $0<x\leq w$ of the right well, where energy is removed only at impacts. The flow removes energy only where $x<-\sqrt\beta$, in the left well. Near the stop, the balance between the energy supplied along the flow and the energy removed at impacts produces the attracting Zeno point and the repelling impact orbit described below. The connecting orbits described below depend on the parameters, and with $\varepsilon$, $w$, and $c$ fixed, they change at $\beta\approx0.7147$ and at $\beta\approx0.8138$.

Here $R:=[-1.95,0.8]\times[-2.35,1.95]$, and $X$ is the closure of the set of states reached from $R$. Note that $r(G\cap R)\subseteq R$, since $c\cdot1.95<2.35$, and that $\mathcal H$ satisfies TGC on $X$. Indeed, the flow crosses $G$ transversally at $(w,v)$ with $v>0$, and at $(w,0)$ it satisfies $\dot v=w-w^3=0.288>0$, so the states of $X$ near $(w,0)$ reach the stop in a time that tends to $0$, at a point that tends to $(w,0)$, as they approach $(w,0)$. The following Liénard function controls the growth of orbits.

### Lemma (`lem:impact-lienard`)

Let $\eta(x):=\int_w^x(y^2-\beta)\,dy$, let $L(x,v):=\frac12\left(v+\varepsilon\eta(x)\right)^2+V(x)$, and let $\lambda:=\max_{x\leq w}\left(-\varepsilon\eta(x)V'(x)\right)$.

- (i) Along the flow, $\dot L=-\varepsilon\eta(x)V'(x)$.
- (ii) An impact at speed $v$ lowers $L$ by $(1-c^2)v^2/2$.
- (iii) Every hybrid trajectory $\psi_z$ satisfies

  $$
  L(\psi_z(t,n))\leq L(z)+\lambda t\quad\forall(t,n)\in\operatorname{dom}(\psi_z).
  $$

- (iv) For every $L_0\in\mathbb R$, the set $\lbrace(x,v)\mid x\leq w,\ L(x,v)\leq L_0\rbrace$ is compact.

Proof. Since $\dot v=-V'(x)-\varepsilon\eta'(x)v$, differentiation along the flow yields

$$
\dot L=\left(v+\varepsilon\eta(x)\right)\left(\dot v+\varepsilon\eta'(x)v\right)+V'(x)v=-\left(v+\varepsilon\eta(x)\right)V'(x)+V'(x)v=-\varepsilon\eta(x)V'(x),
$$

which proves (i). Since $\eta(w)=0$, the function $L$ agrees with $v^2/2+V(w)$ on the stop, so an impact at speed $v$ lowers $L$ by $(v^2-c^2v^2)/2$, which proves (ii). The maximum $\lambda$ exists because $-\varepsilon\eta(x)V'(x)\to-\infty$ as $x\to-\infty$. Integrating $\dot L\leq\lambda$ between impacts and using (ii) at the impacts yields (iii). Finally, the set in (iv) is closed, since $L$ is continuous. Moreover, $L(x,v)\leq L_0$ implies $V(x)\leq L_0$, which bounds $x$, and then $|v|\leq\varepsilon|\eta(x)|+\sqrt{2L_0+1/2}$, since $V\geq-1/4$. Hence the set is compact. $\square$

The lemma shows that the flow has no blow-up and that every hybrid trajectory remains in a compact set over every bounded time interval.

### Numerical isolation

The following statements are numerical, not proved.

- The constant of the lemma is $\lambda\approx0.8491$, attained at $x\approx-1.489$ (a numerical maximum). A recomputation on a grid of $x$ gives $\lambda=0.84914$ at $x=-1.48941$.
- $\widetilde D=\Sigma(R)$ is not forward invariant, since the states on the edge $x=-1.95$ with $v<0$ leave $R$. Numerically, $\widetilde D$ is nevertheless an attracting neighborhood of $\Phi$. The manuscript followed $20{,}000$ states on each edge of $R$ and a grid of $398\times398$ states inside $R$, and then searched near the states that stay outside $\operatorname{int}_{\Sigma X}\widetilde D$ longest. The latest time at which one of these trajectories lies outside $\operatorname{int}_{\Sigma X}\widetilde D$ is about $3.748$. These samples indicate that $\Phi([T,\infty),\widetilde D)\subseteq\operatorname{int}_{\Sigma X}\widetilde D$ with $T=3.75$, but this inclusion is not proved.
- If the inclusion holds, then $X$ is compact, since the states reached from $R$ after time $T$ lie in $R$, and by `lem:impact-lienard`(iii) and (iv) those reached before time $T$ lie in the compact set $\lbrace L\leq\max_RL+\lambda T\rbrace\subseteq\lbrace L\leq9.2\rbrace$. On a grid of $R$, $\max_RL\approx5.925$, attained at the corner $(-1.95,-2.35)$, so $\max_RL+\lambda T\approx9.109$. `lem:window-isolation` then yields $\operatorname{Inv}(\widetilde D,f_\tau)=\omega(\widetilde D,\Phi)\subseteq\operatorname{int}_{\Sigma X}\widetilde D$ for every $\tau>0$.

### Invariant sets and indices

The equilibria are the stable focus $Q:=(-1,0)$, with eigenvalues $-0.12\pm i\sqrt{1.9856}$, and the saddle $S:=(0,0)$, with eigenvalues $0.38\pm\sqrt{1.1444}$, that is, $1.450$ and $-0.690$. The stop contains the Zeno point $Z:=(w,0)$, which corresponds to the periodic orbit $\widetilde Z:=\Sigma(\lbrace Z\rbrace)$ of $\Phi$. Physically, the orbits that converge to $Z$ describe a particle that comes to rest against the stop after infinitely many impacts in finite time.

Numerically, the recurrent dynamics in $R$ consists of these three sets and two periodic orbits with one impact per period. The first, $C$, is attracting and oscillates across both wells, with impact speed $1.50984$, multiplier $0.2887$ of the return map on the guard, and period $5.956$ under $\Phi$. The second, $U_Z$, is repelling and surrounds $Z$, with impact speed $0.65696$, multiplier $2.229$, and period $4.168$. Numerically, a bounce whose impact speed is smaller than that of $U_Z$ loses more energy at its impact than it gains along the flow before the next impact, and a bounce whose impact speed is slightly larger gains more than it loses. The right branch of the unstable manifold of $S$ first reaches the stop at speed $0.8997$, between the impact speeds of $U_Z$ and $C$, and converges to $C$, while the left branch converges to $Q$. Both branches of the stable manifold of $S$ accumulate on $U_Z$ in backward time. The orbits that leave $U_Z$ on the inside converge to $Z$, and those that leave it on the outside converge to $C$, to $Q$, or, along the stable manifold of $S$, to $S$. Hence, after transitive reduction, the connecting orbits order the five sets by $U_Z\to Z$, $U_Z\to S$, $S\to C$, and $S\to Q$. Of a $400\times400$ grid of states in $R$, $81.2\%$ converge to $C$, $16.1\%$ to $Q$, and $2.67\%$ to $Z$.

Numerically, therefore, $C$, $Q$, and $Z$ are attractors of $\mathcal H$, and $\Sigma(C)$, $\lbrace\tilde\iota(Q)\rbrace$, $\widetilde Z$, $\lbrace\tilde\iota(S)\rbrace$, and $\Sigma(U_Z)$ are the Morse sets of a Morse representation of $\omega(\widetilde D,\Phi)$ with this order. The attractors of $\Phi$ in $\omega(\widetilde D,\Phi)$ then correspond to the eleven down-sets of the order. Among them are $\Sigma(C)$, $\lbrace\tilde\iota(Q)\rbrace$, $\widetilde Z$, and their unions. The indices of the attractors are $\Sigma\mathrm{CH}_{*}(\lbrace Q\rbrace)\cong H_{*}(\mathrm{pt})$ and $\Sigma\mathrm{CH}_{*}(C)\cong\Sigma\mathrm{CH}_{*}(\lbrace Z\rbrace)\cong H_{*}(S^1)$, since $\Sigma(C)$ and $\widetilde Z$ are attracting periodic orbits of $\Phi$, as for the bouncing ball. Since $S$ is a hyperbolic saddle with one unstable direction, $\Sigma\mathrm{CH}_k(\lbrace S\rbrace)\cong\mathbb F$ for $k=1$ and $\Sigma\mathrm{CH}_k(\lbrace S\rbrace)=0$ otherwise. Near $\Sigma(U_Z)$ the space $\Sigma X$ is a surface, and $\Sigma(U_Z)$ is a repelling periodic orbit with positive multiplier, so $\Sigma\mathrm{CH}_k(U_Z)\cong\mathbb F$ for $k=1,2$ and $\Sigma\mathrm{CH}_k(U_Z)=0$ otherwise.

These indices follow from the numerical description of the five sets and their stability, so they are as reliable as that description. `PAPER_GRID.md`, "The oscillator at `beta = 0.76`", gives the extents of the two cycles: $C$ has $x\in[-1.6859,0.8]$ and $v\in[-1.8769,1.5098]$, and $U_Z$ has $x\in[0.3761,0.8]$ and $v\in[-0.4599,0.6570]$.

### Comparison with the run

The run uses $\tau=0.5$ on $\Xi_7$, with 2048 base cells per axis and 512 phase intervals. Since $\widetilde Z$ has period $1$, $f_1$ is the identity on $\widetilde Z$, while $f_{0.5}$ moves each point of $\widetilde Z$ by half the period. The Morse graph has 22 Morse sets and 24 edges. The runner located points of the five sets in the Morse sets (`reference_set_identification`; the JSON key of $Q$ is `F`):

| Set | Points located (base, handle) | Morse set | Points in no Morse set |
|---|---|---|---|
| $C$ | 400, 400 | $M(0)$ (all 800) | 0 |
| $Q$ | 1, 0 | $M(1)$ | 0 |
| $Z$ | 1, 400 | $M(2)$ (all 401) | 0 |
| $S$ | 1, 0 | $M(11)$ | 0 |
| $U_Z$ | 400, 400 | $M(21)$ (all 800) | 0 |

The labels agree with the indices above.

| Set | Index | Morse set | Cells | Label | Homology of the pair |
|---|---|---|---|---|---|
| $C$ | $H_{*}(S^1)$ | $M(0)$ | 192,152 base cells in $[-1.703,0.8]\times[-1.934,1.562]$; 48,231 handle pieces meeting all 512 phase intervals | $(x-1,x-1,0,0)$ | $(1,1,0,0)$ |
| $Q$ | $H_{*}(\mathrm{pt})$ | $M(1)$ | 3,868 base cells in $[-1.049,-0.9483]\times[-0.07192,0.06875]$; no handle piece | $(x-1,0,0,0)$ | $(1,0,0,0)$ |
| $Z$ | $H_{*}(S^1)$ | $M(2)$ | 141 base cells in $[0.7946,0.8]\times[-0.04043,0.04985]$; 25,361 handle pieces meeting all 512 phase intervals | $(x-1,x-1,0,0)$ | $(1,2,0,0)$ |
| $S$ | $\mathbb F$ in degree 1 | $M(11)$ | 58 base cells in $[-0.01104,0.01045]\times[-0.008936,0.009961]$; no handle piece | $(0,x-1,0,0)$ | $(0,1,0,0)$ |
| $U_Z$ | $\mathbb F$ in degrees 1 and 2 | $M(21)$ | 40,973 base cells in $[0.3193,0.8]\times[-0.4813,0.6986]$; 33,501 handle pieces meeting all 512 phase intervals | $(0,x-1,x-1,0)$ | $(0,1,1,0)$ |

- $Z$: the label has the form of $H_{*}(S^1)$ although $\dim H_1=2$, as for the bouncing ball.
- $C$: the pair has 240,383 pieces, and its label was computed at `a47c88d` without a piece limit.
- $U_Z$: the label comes from the excision pair (`label_source: "index map (excision pair)"`). The index map on the pair above failed its carrier check. The set $\mathcal F(M(21))\setminus M(21)$ consists of two annuli, and since the carrier sends a piece of this set to all pieces of its component, the carrier of a single piece is not acyclic over $\mathbb F_5$. `demo/fill_missing_labels.py` with `index_map="auto"` then used the excision pair, with $\bar X$ of 84,612 atoms (104,212 pieces) and $\bar A$ of 23,842 atoms (29,738 pieces), relative homology of dimensions $(0,1,1,0)$, and index map $(1)$ in degrees $1$ and $2$. The fill was committed in `02270ea`. It checked the recomputed relation (75,853,808 edges), Morse sets, and Morse graph against the record, and it changed no other index record.

Order. The three minimal Morse sets are $M(0)$, $M(1)$, and $M(2)$, which contain the attractors $C$, $Q$, and $Z$. The figure `<stem>-nontrivial.pdf` shows exactly $M(0)$, $M(1)$, $M(2)$, $M(11)$, and $M(21)$, with the edges $M(21)\to M(2)$, $M(21)\to M(11)$, $M(11)\to M(0)$, and $M(11)\to M(1)$, which is the order $U_Z\to Z$, $U_Z\to S$, $S\to C$, $S\to Q$. In the full Morse graph, $M(21)\to M(2)$ is an edge, $M(21)$ reaches $M(11)$ through $M(17)$, $M(14)$, $M(12)$ and through $M(20)$, $M(19)$, then $M(15)$ or $M(18)$, $M(16)$, and then $M(13)$, and $M(11)$ reaches $M(0)$ through $M(8)$, $M(5)$ and through $M(10)$, $M(7)$, $M(4)$, and reaches $M(1)$ through $M(9)$, $M(6)$, $M(3)$. Every path from $M(21)$ to $M(0)$ or $M(1)$ passes through $M(11)$.

The other 17 Morse sets contain none of the five sets. Their pairs have zero relative homology, so their labels are $(0,0,0,0)$, and the figure `<stem>-nontrivial.pdf` hides them. Write a point near $S$ as $S+u\,e_u+s\,e_s$ with the unstable and stable eigenvectors $e_u=(1,1.450)$ and $e_s=(1,-0.690)$. Sixteen of the 17 sets have no handle piece and lie within $0.015$ of $S$. By their cell centers they fall into the first three groups below, and $M(3)$ lies around $Q$.

- $u>|s|$, along the right branch of the unstable manifold of $S$, toward $C$: $M(4)$, $M(5)$, $M(7)$, $M(8)$, $M(10)$, single base cells within $0.0074$ of $S$.
- $-u>|s|$, along the left branch of the unstable manifold, toward $Q$: $M(6)$, $M(9)$, single base cells within $0.0065$ of $S$.
- $|s|>4|u|$, along the two branches of the stable manifold, which accumulate on $U_Z$ in backward time: $M(12)$ (2 base cells), $M(14)$, $M(17)$ with $s<0$, and $M(13)$ (3 base cells), $M(15)$, $M(16)$, $M(18)$, $M(19)$, $M(20)$ with $s>0$. They lie between $M(21)$ and $M(11)$ in the order.
- $M(3)$: 38 base cells in 12 components in $[-1.049,-0.9496]\times[-0.07192,0.07085]$, around $Q$, between $M(6)$ and $M(1)$ in the order. Its pair has 317 atoms, 279 of them in $\mathcal F(M)\setminus M$.

The figure `<stem>.pdf` has three zooms: A shows $M(3)$ around $Q$, B shows $M(11)$ and the 16 Morse sets near $S$, and C shows $M(2)$ at the stop. The figure `<stem>-nontrivial.pdf` has two: A shows $M(11)$ and B shows $M(2)$.

## Status of each statement

| Statement | Status |
|---|---|
| `lem:window-isolation` | proved in the earlier draft; uses compactness of $X$ and continuity of $\Phi$ |
| `prop:ball-window` | proved in the earlier draft |
| `prop:wheel-isolation` | proved in the earlier draft; uses neither compactness of $X$ nor continuity of $\Phi$ |
| `lem:impact-lienard` | proved in the earlier draft; the value $\lambda\approx0.8491$ is a numerical maximum |
| oscillator: $\Phi([3.75,\infty),\widetilde D)\subseteq\operatorname{int}_{\Sigma X}\widetilde D$, hence compactness of $X$ and isolation of $\widetilde D$ | numerical, from sampling; not proved |
| oscillator: $C$, $U_Z$, their impact speeds, multipliers, and periods, the order of the five sets, the basin fractions | numerical |
| oscillator: indices of $C$, $Q$, $Z$, $S$, $U_Z$ | follow from the numerical description of the five sets |
| ball: $\Sigma\mathrm{CH}_{*}(\lbrace(0,0)\rbrace)\cong H_{*}(S^1)$; neuron: $\Sigma\mathrm{CH}_{*}(A)\cong H_{*}(S^1)$ | recalled from [`Rivas:Kalies`] |
| wheel: indices of the gait and the saddle | local comparison with [`Rivas:Kalies`], since $X$ is not compact |
| computed Morse graphs and labels | sampled multivalued maps, not certified outer approximations; consistency checks only |
