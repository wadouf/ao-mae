# OA-MAE scientific contract

## External cloud support

For token j and time t:

\[
\bar c^t(j)=\frac{1}{16^2}\sum_{p\in R_j}C^t(p),
\]

\[
c'^t(j)=\min\left(1,\bar c^t(j)+0.30\bar c^t(j)\mathbb I[\bar c^t(j)\geq0.20]\right).
\]

Broadcast token values to pixels and compute:

\[
M^t(p)=\mathbb I[c'^t(p)\leq0.85],\qquad
V_{12}=M^{T_1}\cap M^{T_2}.
\]

The support is external, method-invariant, and shared by all compared methods.

## Stage I fusion

\[
g_{cloud}^t(j)=\sigma(10(c'^t(j)-0.50)),
\]

\[
r_{sar}^t(j)=\sigma(\theta_{struc}\tilde{s}^t(j)-\theta_{noise}\tilde{\nu}^t(j)+\theta_{bias}),
\]

\[
g^t(j)=g_{cloud}^t(j)r_{sar}^t(j).
\]

Use optical queries and projected SAR keys and values in the final four optical blocks. The optical stream remains the primary representation.

## Cloud-Mix

\[
\Omega=\Omega_{cloud}\cup\Omega_{struct}\cup\Omega_{rand},\qquad |\Omega|=0.75N_{tok}.
\]

Masking occurs before optical encoding.

## Past-only targets

\[
\mathcal H_{<t,90}=\{\tau<t:0<t-\tau\leq90\text{ days}\}.
\]

Retain the Top-3 clearest candidates per token and compute:

\[
X^{\star,t}(j)=\operatorname{med}_{\tau\in\mathcal T_j^t}X_{opt}^{\tau}(j).
\]

No candidate may occur at or after t.

## Target safety

\[
d_{sar}^t(j)=\left\|\phi(S^t(j))-\operatorname{med}_{\tau\in\mathcal T_j^t}\phi(S^\tau(j))\right\|_2,
\]

\[
\omega_{safe}^t(j)=\max(0.10,1-10d_{sar}^t(j)),
\]

\[
\omega_{clamp}^t(j)=\mathbb I[c'^t(j)\leq0.85],\qquad
w^t(j)=\omega_{safe}^t(j)\omega_{clamp}^t(j).
\]

## Stage I objective

\[
\mathcal L_{pre}=1.00\mathcal L_{rec}+0.50\mathcal L_{str}+0.10\mathcal L_{rr}.
\]

Structural fallback is used only when no valid optical target exists.

## Stage II

\[
F_{\Delta}=\operatorname{Conv}_{1\times1}\left(\operatorname{Concat}[|F_{T_2}-F_{T_1}|,F_{T_2}\odot F_{T_1}]\right).
\]

The decoder produces \(\hat Y\in[0,1]^{H\times W}\), and \(\hat Y_b=\mathbb I[\hat Y\geq0.50]\).

The supervised loss is Focal plus Dice on V12. V12 does not enter the decoder or probability-map computation.

## Operational output

\[
O(p)=\begin{cases}
\text{Change}, & p\in V_{12},\hat Y_b(p)=1,\\
\text{No change}, & p\in V_{12},\hat Y_b(p)=0,\\
\text{Unresolved}, & p\notin V_{12}.
\end{cases}
\]

Human review, SAR-only fallback, and reacquisition are separate follow-up policies, not native OA-MAE inference.
