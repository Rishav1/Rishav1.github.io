---
layout: distill
mathjax: true
title: "DP Dynamics of Noisy Gradient Descent (Part 2)"
date: 2021-11-22 10:00:00 +0700
categories: [Research]
tags: [Differential Privacy, Noisy Gradient Descent]
bibliography: /2021-11-22-DPDynamics.bib
authors:
  - name: Rishav Chourasia*
    url: https://comp.nus.edu.sg/~rishav1
    affiliations:
      name: NUS, Singapore
  - name: Jiayuan Ye*
    url: https://comp.nus.edu.sg/~jiayuan
    # affiliations:
    #   name: NUS, Singapore
  - name: Reza Shokri
    url: https://comp.nus.edu.sg/~reza
    # affiliations:
    #   name: NUS, Singapore
toc:
  - name: Introduction
  - name: Problem Formulation
  - name: Privacy Analysis
  - subsections:
      - name: Tracing Diffusion for Noisy GD
      - name: Privacy erosion along tracing diffusion
      - name: Controlling Rényi privacy loss rate
  - name: Tightness Analysis
  - name: Utility Analysis
  - name: Conclusion
---

<p style="display: none;">
$$
\newcommand{\D}{D}
\newcommand{\smh}{\beta}
\newcommand{\loss}[2]{\ell(#2;\mathbf{#1})}
\newcommand{\ptheta}{\theta}
\newcommand{\thet}{\Theta}
\newcommand{\x}{\mathbf{x}}
\newcommand{\K}{K}
\newcommand{\T}{T}
\newcommand{\kk}{k}
\newcommand{\proj}[2]{\Pi_{#1}(#2)}
\newcommand{\C}{\mathcal{C}}
\newcommand{\sig}{\sigma}
\newcommand{\q}{\alpha}
\newcommand{\eps}{\varepsilon}
\newcommand{\size}{n}
\newcommand{\step}{\eta}
\newcommand{\R}{\mathbb{R}}
\newcommand{\Univ}{\mathcal{X}}
\newcommand{\thetaspace}{\mathbb{R}^d}
\newcommand{\ve}{\mathbf v}
\newcommand{\Loss}{\mathcal{L}}
\newcommand{\Domain}{\Univ^\size}
\newcommand{\Ren}[3]{R_{#1}\left(#2\middle\|#3\right)}
\newcommand{\deltW}[1]{\dif{\W{#1}}}
\newcommand{\graD}[1]{\nabla #1}
\newcommand{\hesS}[1]{\nabla^2 #1}
\newcommand{\tra}[1]{\textsf{Tr}\left(#1\right)}
\newcommand{\dif}[1]{d #1}
\newcommand{\der}[2]{\frac{\dif{#1}}{\dif{#2}}}
\newcommand{\doh}[2]{\frac{\partial #1}{\partial #2}}
\newcommand{\lapL}[1]{\Delta #1}
\newcommand{\divR}[1]{\nabla \cdot #1}
\newcommand{\W}[1]{\mathbf{W}_{#1}}
\newcommand{\E}{\mathbb{E}}
\newcommand{\Expec}[2]{\underset{#1}{\E}\left[#2\right]}
\newcommand{\norm}[1]{\left\lVert#1\right\rVert_2}
\newcommand{\Fren}[3]{E_{#1}\left(#2\middle\|#3\right)}
\newcommand{\Gren}[3]{I_{#1}\left(#2\middle\|#3\right)}
\newcommand{\sen}[1]{S_{#1}}
$$
</p>

What is the information leakage of an iterative randomized learning algorithm about its training data, when the internal state of the algorithm is private? How much is the contribution of each specific training epoch to the information leakage through the released model? We study this problem for noisy gradient descent algorithms, and model the dynamics of Rényi differential privacy loss throughout the training process. Our analysis traces a provably tight bound on the Rényi divergence between the pair of probability distributions over parameters of models trained on neighboring datasets. We prove that the privacy loss converges exponentially fast, for smooth and strongly convex loss functions, which is a significant improvement over composition theorems (which over-estimate the privacy loss by upper-bounding its total value over all intermediate gradient computations). For Lipschitz, smooth, and strongly convex loss functions, we prove optimal utility with a small gradient complexity for noisy gradient descent algorithms. We provide additional details about our analysis in the [full version >](https://arxiv.org/abs/2102.05855.pdf).
{: .text-justify}

## Introduction

<!-- <d-cite key=""></d-cite> -->

Machine learning models leak a significant amount of information about their training data, through their parameters and predictions<d-cite key="shokri2017membership, nasr2019comprehensive, carlini2020extracting"></d-cite>. Iterative randomized training algorithms can limit this information leakage and bound the differential privacy loss of the learning process <d-cite key="bassily2014private, abadi2016deep, feldman2018privacy, feldman2020private"></d-cite>. The strength of this certified defense is determined by an _upper bound_ on the (Rényi) divergence between the probability distributions of model parameters learned on any pair of neighboring datasets.
{: .text-justify}

The general method to compute the differential privacy bound for gradient perturbation-based learning algorithms is to view the process as a number of (identical) differential privacy mechanisms, and to compute the _composition_ of their bounds. However, this over-estimates the privacy loss of the released model<d-cite key="jagielski2020auditing, nasr2021adversary"></d-cite>, and results in a loose differential privacy bound. This is because composition bounds also accounts for the leakage of all intermediate gradient updates, even though only the final model parameters are observable to adversary. Feldman et al.<d-cite key="feldman2018privacy, feldman2020private"></d-cite> address this issue for the privacy analysis of gradient computations over _one single_ training epoch, for smooth and convex loss functions. However, in learning a model over multiple training epochs, such a guarantee is quantitatively similar to composition bounds of privacy amplification by sub-sampling<d-cite key="feldman2018privacy"></d-cite>. The open challenge, that we tackle in this paper, is to provide an analysis that can tightly bound the privacy loss of the _released model_ after $K$ training epochs, for any $K$.
{: .text-justify}

We present a novel analysis for privacy dynamics of noisy gradient descent with smooth and strongly convex loss functions. We construct a pair of continuous-time Langevin diffusion<d-cite key="sato2014approximation"></d-cite> processes that trace the probability distributions over the model parameters of noisy GD. Subsequently, we derive differential inequalities bounding the _rate of privacy loss_ (worst case Rényi divergence between the **coupled stochastic processes** associated with neighboring datasets) throughout the training process. Using it, we prove an exponentially-fast converging privacy bound for noisy GD:
{: .text-justify}

<div id="main_theorem" class="theorem"><b>1.</b>
Under $\lambda$-strongly convex and $\smh$-smooth loss function $\loss{\x}{\ptheta}$ with gradient sensitivity $\sen{g}$, the <a href="#goal-noisygd">noisy GD Algorithm 1</a> for $\K$ steps, with initial parameter vector $\ptheta_0\sim\proj{\C}{\mathcal{N}(0,2\sig^2\mathbb{I}_d)}$, step size $\step < \frac{1}{\smh}$, and noise variance $\sig^2$, satisfies $(\q, \eps)$-Rényi DP with $\eps = \frac{\q\sen{g}^2}{\lambda\sigma^2\size^2}(1-e^{-\step\frac{K}{2}})$, where $n$ is the size of the training set.
</div>
<aside markdown="1">
Here $\C$ is any convex set in $\R^d$ and the projection operation is defined as $\proj{\C}{\ptheta} \triangleq \underset{\ptheta' \in \C}{\arg \min} \lVert \ptheta - \ptheta' \rVert$.
</aside>

This guarantee shows that the privacy loss **converges** exponentially in the number of iterations $K$, instead of growing proportionally with $K$ as in the composition-based analysis of the same algorithms. Our bound captures a strong privacy amplification due to the dynamics (and convergence) of differential privacy over the noisy gradient descent algorithm with private internal state.
{: .text-justify}

<!-- #####  -->

<div class="remark">
We also prove that this <b>bound is tight</b> by showing that there exist a loss function and neighboring datasets such that the divergence between corresponding model parameter distributions grows at a matching order. As a consequence of this improved privacy bound, we prove that noisy GD achieves <b>optimal utility</b> under differential privacy with an error of order $O(\frac{d}{n^2\eps^2})$, with a <i>small gradient complexity</i> of order $O(n\log(n))$. This is a significant improvement over the prior utility analysis for noisy SGD algorithms<d-cite key="bassily2014private"></d-cite>, wherein the gradient complexity reduces by a factor of ${\size/\log(\size)}$, and attainable utility improves by a factor of $\text{poly}\log(\size)$. 
</div>
{: .text-justify}

## Problem Formulation

Let $\D = (\x_1, \x_2, \cdots, \x_\size)$ be a dataset of size $\size$ with records taken from a universe $\Univ$. For a given machine learning algorithm, let ${\loss{\x}{\ptheta} : \Univ \times \thetaspace \rightarrow \R}$ be a loss function of a parameter vector $\ptheta \in \C$ on the data point $\x$, where $\C$ is a closed convex set (can be $\mathbb{R}^d$).
{: .text-justify}

A generic formulation of the optimization problem to learn the model parameters, is in the form of empirical risk minimization (ERM) with the following objective, where $\Loss_\D(\ptheta)$ is the empirical loss of the model, with parameter vector $\ptheta$, on a dataset $\D$.
{: .text-justify}

$$
\ptheta^* = \underset{\ptheta \in \C}{\arg \min}~\Loss_\D(\ptheta), \quad \text{where} \quad \Loss_\D(\ptheta)=\frac{1}{\size} \sum_{\x \in \D}\loss{\x}{\ptheta}.
$$

Releasing this optimization output (i.e., $\ptheta^*$) can leak information about the dataset $\D$, hence violating data privacy. To mitigate this risk, there exist randomized algorithms to ensure that the ($\q$-Rényi) privacy loss of the ERM algorithm is upper-bounded by $\eps$, i.e., the algorithm satisfies ${(\q, \eps)\text{-RDP}}$.
{: .text-justify}

<aside markdown="1">
A randomized algorithm $\mathcal A: \Domain \rightarrow \thetaspace$ satisfies <b>$(\q, \eps)$-Rényi Differential Privacy</b> (RDP), if for any two neighboring datasets $\D, \D' \in \Domain$, the $\alpha$-Rényi divergence is bounded: 
$\Ren{\q}{\mathcal A(\D)}{\mathcal A(\D')} \leq \eps$.
</aside>

<figure id="renyi_differential_privacy">
<picture>
    <source srcset="{{ site.baseurl }}/assets/img/blog/DPDynamics/renyi_differential_privacy.png" media="(min-width: 768px)" />
    <img srcset="renyi_differential_privacy.png" alt="RDP" />
</picture>
</figure>

In this paper, our objective is to model and analyze the **dynamics of differential privacy** of a popular randomized ERM algorithm called _Noisy Gradient Descent_<d-cite key="bassily2014private"></d-cite> (<a href="#goal-noisygd">Algorithm 1</a>).
{: .text-justify}

<pre id="read-noisygd" style="display:none;">
\begin{algorithm}
\caption{$\mathcal A_{\text{Noisy-GD}}$: Noisy Gradient Descent}
\begin{algorithmic}
\PROCEDURE{Noisy-GD}{Dataset $D = (\mathbf{x}_1, \mathbf{x}_2, \cdots, \mathbf{x}_n)$, loss function $\ell(\theta;\mathbf{x})$, closed convex set $\mathcal{C}\subseteq\mathbb{R}^d$, learning rate $\eta$, noise variance $\sigma^2$, initial parameter vector $\theta$}
    \FOR{$k = 0, 1, \cdots, K - 1$} 
        \STATE {$g(\theta_k;D)= \sum_{i=1}^n \nabla\ell({\theta_k};{\mathbf{x}_i})$}
        \STATE {$\theta_{k+1} = \Pi_{\mathcal C} (\theta_{k} - \frac{\eta}{n} g(\theta_k;D) + \sqrt{2\eta} \mathcal{N}(0, \sigma^2 \mathbb{I}_d))$}
    \ENDFOR
    \STATE {Output $\theta_{K}$}
\ENDPROCEDURE
\end{algorithmic}
\end{algorithm}
</pre>
<div id="goal-noisygd"></div>
<script type="text/javascript">
    var code = document.getElementById("read-noisygd").textContent;
    var parentEl = document.getElementById("goal-noisygd");
    var options = {
        lineNumber: true
    };
    pseudocode.render(code, parentEl, options);
</script>

More precisely, we focus on the following.

- Compute an RDP bound (i.e., the worst case Rényi divergence $\Ren{\q}{\Theta_K}{\Theta_K'}$ between the output distributions of two neighboring datasets) for <a href="#goal-noisygd">Algorithm 1</a>, and analyze its tightness.
- Compute the contribution of each iteration to the privacy loss. As we go from step $k = 1$ to $K$ in <a href="#goal-noisygd">Algorithm 1</a>, we investigate how the algorithm's privacy loss changes as it runs the $k$'th iteration (computed as ${\Ren{\q}{\Theta_{k}}{\Theta_{k}'} - \Ren{\q}{\Theta_{(k-1)}}{\Theta_{(k-1)}'}}$).

We emphasize that our goal is to construct a theoretical framework for analyzing privacy loss of releasing the output $\ptheta_K$ of the algorithm, assuming _private_ internal states (i.e., $\ptheta_1, \cdots, \ptheta_{K-1}$). In the end, we aim to provide a RDP bound that is tight, thus facilitating optimal empirical risk minimization with differential privacy<d-cite key="bassily2014private"></d-cite>.
{: .text-justify}

## Privacy Analysis

### Tracing Diffusion for Noisy GD

To analyze the privacy loss of <a href="#goal-noisygd">noisy GD</a>, which is a _discrete-time stochastic process_, we first interpolate each discrete update from $\ptheta_k$ to $\ptheta_{k+1}$ with a piece-wise continuously differentiable diffusion process. Let $\D$ and $\D'$ be a pair of arbitrarily chosen neighboring datasets. Given step-size $\step$ and initial parameter vector $\ptheta_0=\ptheta_0'$, the respective $k$'th discrete updates in <a href="#goal-noisygd">Algorithm 1</a> on neighboring datasets $D$ and $D'$ are
{: .text-justify}

$$
\begin{aligned}
\ptheta_{k+1} &= \proj{\C}{\ptheta_{k} - \step \nabla{\Loss_\D(\ptheta_k)} + \sqrt{2\step\sig^2} Z_k},\\\\
\ptheta_{k+1}' &= \proj{\C}{\ptheta_{k}'-\step\nabla{\Loss_\D(\ptheta_k')}+ \sqrt{2\step\sig^2} Z_k },
\end{aligned}
$$

with $Z_k \sim \mathcal{N}(0,\mathbb{I}_d)$.

These two discrete jumps can be interpolated with two stochastic processes $\thet_t$ and $\thet_t'$ over time $\step k \leq t\leq\step (k+1)$ respectively (visualized below via dotted lines).
{: .text-justify}

<figure id="coupled_diffusion">
<picture>
    <source srcset="{{ site.baseurl }}/assets/img/blog/DPDynamics/coupled_diffusion.png" media="(min-width: 768px)" />
    <img srcset="coupled_diffusion.png" alt="RDP" />
</picture>
</figure>

At the start of each step, $t=\eta k$, the random variables $\thet_{\eta k}$ and $\thet_{\eta k}'$ model the distribution of the $\theta_k$ and $\theta_k'$ in the noisy GD processes respectively. Immediately after $t=\eta k$, the random variables undergo identical mapping $\phi(\cdot)$ defined as $\phi(\theta) = \theta - \eta U_+(\theta)$. During time $\eta k< t <\eta (k+1)$, the two process diffuse along opposing vector fields ($U_-(\cdot)$ and $-U_-(\cdot)$) with Brownian perturbation. At the end of step, i.e. at $t \rightarrow \eta (k+1)$, we project $\thet_t$ and $\thet_t'$ onto convex set $\C$, and obtain $\thet_{\eta (k + 1)}$ and $\thet_{\eta (k + 1)}'$.
{: .text-justify}

Repeating the construction for $k=0,\cdots,K-1$, we define two piece-wise continuous diffusion processes $\\{ \thet_t \\}\_{t \geq 0}$ and $\\{ \thet_t' \\}\_{t \geq 0}$ whose distributions at time $t=\step k$ are consistent with $\theta_k$ and $\theta_{k}'$ in the noisy GD processes for any $k \in \\{0,\cdots, K-1\\}$.
{: .text-justify}

<div class="definition"><b>(Coupled tracing diffusions).</b> 
Let $\thet_0=\thet_0'$ be two identically distributed random variables. We refer to the stochastic processes $\{ \thet_t \}_{t \geq 0}$ and $\{ \thet_t' \}_{t \geq 0}$ that evolve along <a href="#coupled_diffusion">diffusion processes</a> in $\eta k< t < \eta (k+1)$ and undergo projection steps $\proj{\C}{\cdot}$ at the end of step $t = \step(k+1)$, as coupled tracing diffusions for noisy GD on neighboring datasets $\D,\D'$.    
</div>
{: .text-justify}

The Rényi divergence $R_{\alpha}(\thet_{\eta \K}\lVert\thet_{\eta \K}')$ reflects the Rényi privacy loss of <a href="#goal-noisygd">noisy GD</a> with $\K$ steps. Conditioned on observing $\ptheta_k$ and $\theta_\kk'$, the processes $\\{\Theta_t\\}\_{\step\kk< t< \step(\kk+1)}$ and $\\{\Theta_t'\\}\_{\step\kk< t<\step(\kk+1)}$ are _Langevin diffusions_ along vector fields $ - U*-(\ptheta*{\kk})$ and $U_-(\ptheta_{\kk}')$ respectively, for duration $\step$. That is, conditioned on observing $\theta_\kk$ and $\theta_\kk'$, the two diffusion processes have the following stochastic differential equations (SDEs) respectively.
{: .text-justify}

$$
\dif{\thet_t} = - U_-(\theta_k)\dif{t} + \sqrt{2\sig^2}\deltW{t}, \quad \dif{\thet_t}' =  U_-(\theta_k')\dif{t} + \sqrt{2\sig^2}\deltW{t},
$$

where $\deltW{t}\sim\sqrt{dt}\mathcal N(0,I_d)$ describe the Wiener processes on $\thetaspace$.

<figure id="renyi_differential_privacy">
    <img src="{{ site.baseurl }}/assets/img/blog/DPDynamics/fokker_planck.gif">
    <figcaption>Visualization of Fokker-Planck equation for Langevin diffusion by <a href="https://twitter.com/gabrielpeyre/status/1433293610586234882">Gabriel Pyeré</a>.</figcaption>
</figure>

The joint effect of the drag force (i.e $- U_-(\theta_k)$) and Brownian fluctuations on the (conditional) probability density $p(\thet_t = \ptheta \mid \thet_{\eta k} = \ptheta_k)$ of random variable $\Theta_t$ given $\Theta_{\eta k} = \ptheta_k$ is characterized through the Fokker-Planck equation<d-cite key="fokker1914mittlere"></d-cite> below. For brevity, we use $p_{t\mid\eta \kk}(\ptheta\mid\ptheta_k)$ and $p_{t\mid\eta \kk}'(\ptheta\mid\ptheta_k')$ to represent the conditional probability density function $p(\thet_t = \ptheta \mid \thet_{\eta k} = \ptheta_k)$ and $p(\thet_t' = \ptheta \mid \thet_{\eta \kk}' = \ptheta_\kk')$ respectively.
{: .text-justify}

$$
\begin{aligned}
\doh{p_{t|\step\kk}(\ptheta|\ptheta_\kk)}{t} &= \divR{\left(p_{t|\step\kk}(\ptheta|\ptheta_\kk) U_-(\theta_k)\right)} + \sig^2 \lapL{p_{t|\step\kk}(\ptheta|\ptheta_\kk)} \\\\
\doh{p_{t | \step \kk}'(\ptheta|\ptheta_\kk')}{t} &= - \divR{\left(p_{t|\step\kk}'(\ptheta|\ptheta_\kk')U_-(\theta_k')\right)} + \sig^2 \lapL{p_{t|\step\kk}'(\ptheta|\ptheta_\kk')}
\end{aligned}
$$

<aside markdown="1">
Given a smooth vector field $\ve :\C \rightarrow \R^d$, its 
<b>Divergence</b> $\divR{\ve}: \C \rightarrow \R$ is $(\divR{\ve})(\ptheta) = \underset{i=1}{\overset{d}{\sum}} \doh{\ve_i(\ptheta)}{\ptheta_i}$.
</aside>

By taking expectations over probability density function $p_{\eta k}(\theta_\kk)$ or $p_{\eta k}'(\theta_\kk')$ on both sides of above equation, we obtain the partial differential equation that models the evolution of (unconditioned) probability density function $p_t(\ptheta)$ and $p_t'(\ptheta)$ in the coupled tracing diffusions.
{: .text-justify}

<aside markdown="1">
Given a smooth function $\Loss:\C \rightarrow \R$, its <b>Laplacian</b> $\lapL{\Loss}:\C \rightarrow \R$ is the trace of its Hessian $\hesS{\Loss}$, (i.e., $\lapL{\Loss(\ptheta)} = \tra{\hesS{\Loss(\ptheta)}}$.
</aside>

<div class="lemma" id="GDupdate_fokker"><b>1.</b> 
For <a href="#coupled_diffusion">coupled tracing diffusion processes</a> in time $\step\kk < t < \step(\kk+1)$, the equivalent Fokker-Planck equations are
$$
\begin{aligned}
\doh{p_t(\ptheta)}{t} =  \divR{(p_t(\ptheta)V_t(\theta))} + \sig^2\lapL{p_t(\ptheta)}, \\
\doh{p_t'(\ptheta)}{t} =  \divR{(p_t'(\ptheta)V_t'(\theta))} + \sig^2\lapL{p_t'(\ptheta)}, 
\end{aligned}
$$
where $V_t(\theta)= \Expec{\ptheta_\kk \sim p_{\step\kk|t}}{U_-(\ptheta_\kk)|\ptheta}$ and $V_t'(\theta) = - \Expec{\ptheta_\kk \sim p_{\step\kk|t}'}{U_-(\ptheta_\kk)|\ptheta}$ are time-dependent vector fields on $\mathbb{R}^d$.
</div>

Since both these vector fields $V_t$ and $V_t'$ are expectations over $U_-(\cdot)$, the loss gradient difference vector between the two databases, it is easy to see that their magnitude is bounded by the sensitivity of total gradient.
{: .text-justify}

$$
\begin{aligned}
\norm{V_t(\ptheta)} &\leq \frac{1}{2} \Expec{\ptheta_k \sim p_{\step k | t}}{\norm{\graD{\Loss_\D(\ptheta_k)} - \graD{\Loss_{\D'}(\ptheta_k)} } | \ptheta} \\\\
&\leq \frac{1}{2\size} \underbrace{\underset{\x, \x' \in \Univ}{\max} \underset{\ptheta \in \thetaspace}{\max}  \norm{\graD{\loss{\x}{\ptheta}} - \graD{\loss{\x'}{\ptheta}}}}_{\text{Sensitivity $\sen{g}$ of loss gradient}}
\end{aligned}
$$

As a result, the magnitude of the vector fields is also bounded, i.e. $\norm{V_t(\ptheta) - V_t'(\ptheta)} \leq \frac{\sen{g}}{\size}$ for all $t \geq 0$ and $\ptheta \in \thetaspace$.
{: .text-justify}

Via these two density evolution equations, we analyze differential privacy dynamics of the noisy gradient descent updates along coupled tracing diffusions.
{: .text-justify}

### Privacy erosion along tracing diffusion

The Rényi divergence (privacy loss) $\Ren{\q}{\thet_t}{\thet_t'}$ between coupled tracing diffusion processes increases over time, as the vector fields $V_t, V_t'$ underlying two processes are different. We refer to this phenomenon as **privacy erosion**. This increase is determined by the amount of change in the probability density functions for coupled tracing diffusions, characterized by the <a href="#GDupdate_fokker">Fokker-Planck equations</a> for diffusions under different vector fields. Using it, we compute a bound on the rate (partial derivative) of $\Ren{\q}{\thet_t}{\thet_t'}$ over time in the following lemma, to model privacy erosion between two different diffusion processes.
{: .text-justify}

<div class="lemma" id="renyi_loss_rate"><b>2 (Rate of Rényi privacy loss).</b> 
Let $V_t$ and $V_t'$ be two vector fields on $\thetaspace$ corresponding to a pair of arbitrarily chosen neighboring datasets $\D$ and $\D'$ of size $\size$, and let $\ell()$ be a loss function with gradient sensitivity $\sen{g}$. Then, for corresponding coupled diffusions $\{\thet_t\}_{t\geq0}$ and $\{\thet_t'\}_{t\geq0}$ under vector fields $V_t$ and $V_t'$ and noise variance $\sig^2$, the Rényi privacy loss rate at any $t\geq0$ is upper bounded by
$$
\doh{\Ren{\q}{\thet_{t}}{\thet_{t}'}}{t} \leq \frac{1}{\gamma}\frac{\q \sen{g}^2}{4\sig^2\size^2} - (1-\gamma)\sig^2\q \frac{\Gren{\q}{\thet_{t}}{\thet_{t}'}}{\Fren{\q}{\thet_{t}}{\thet_{t}'}},
$$
where $\gamma>0$ is a tuning parameter that we can fix arbitrarily according to our need, $\Gren{\q}{\thet_{t}}{\thet_{t}'}$ is the <b>Rényi information</b><d-cite key="vempala2019rapid"></d-cite>, and $\Fren{\q}{\thet_{t}}{\thet_{t}'}$ is the <b>$\alpha$-moment of the likelihood ratio</b><d-cite key="abadi2016deep"></d-cite> between $\thet_{t}$ and $\thet_{t}'$.
</div>

Lemma 2 bounds the rate of privacy loss with various terms. Generally speaking, the term $\frac{\sen{g}^2}{4\sigma^2\size^2}$ bounds the worst-case privacy loss growth due to noisy gradient update, while the term $\frac{I_{\alpha}(\Theta_t\lVert\Theta_t')}{E_{\alpha}(\Theta_t\lVert\Theta_t')}$ amplifies our bound for the rate of privacy loss, as both $\Gren{\q}{\thet_{t}}{\thet_{t}'}$ and $\Fren{\q}{\thet_{t}}{\thet_{t}'}$ are always non-negative. Following is the breakdown of the terms in Lemma 2:
{: .text-justify}

- $\frac{S_g^2}{4\sigma^2n^2}$: This is the first term in the right-hand side of <a href="#renyi_loss_rate">our privacy loss rate bound</a>. It quantifies the worst-case privacy loss due of one noisy gradient update in <a href="#goal-noisygd">noisy GD Algorithm</a>. The term $\frac{S_g}{\size}$ is the sensitivity of average loss gradient $\Loss_D(\theta)$ over two neighboring datasets $D,D'$. The larger $S_g$ is, the further apart the parameters after the gradient descent updates on $D,D'$ could be. The term $\sigma^2$ is the variance of Gaussian noise. Because additive noise shrinks the expected trajectory difference in noisy GD updates, the larger $\sigma^2$ is, the more indistinguishable the distributions are, and therefore, the smaller the privacy loss (Rényi divergence) is.
  {: .text-justify}

- $\frac{I_{\alpha}(\Theta_t\lVert\Theta_t')}{E_{\alpha}(\Theta_t\lVert\Theta_t')}$: This term is the second term on the right-hand side of <a href="#renyi_loss_rate">our privacy loss rate bound</a>, which originates from the derivative of $p_t,p_t'$ with regard to time $t$. The expression $I_{\alpha}/E_{\alpha}$ originates when we use the Fokker Planck equation to substitute the terms related to $\frac{\partial p_t}{\partial t}, \frac{\partial p_t'}{\partial t}$ in the partial derivative of $R_{\alpha}(\Theta_t\lVert\Theta_t')$ with respect to $t$.
  {: .text-justify}

- $\gamma$ is a tuning constant to balance the privacy growth rate estimated using the above two terms, thus helping us tune the privacy loss accumulation.
  {: .text-justify}

Since the ratio $I_\q/E_\q$ is always positive by definition, the <a href="#renyi_loss_rate">Rényi divergence (privacy loss) rate</a> is bounded by its first component (a constant) given any fixed $\q$. Therefore, this naïve privacy analysis gives linear RDP guarantee for Langevin diffusion, which resembles the moment accountant analysis<d-cite key="abadi2016deep"></d-cite>. However, a tighter bound of Rényi privacy loss is possible with finer control of the ratio $I_\q/E_\q$.
{: .text-justify}

### Controlling Rényi privacy loss rate

Although <a href="#renyi_loss_rate">Lemma 2</a> bounds the Rényi privacy loss rate, the Rényi Information term $\Gren{\q}{\thet_{t}}{\thet_{t}'}$ depends on unknown distributions $\thet_{t}, \thet_{t}'$, and is intractable to control in general. Even with explicit expressions for distributions $\thet_{t}, \thet_{t}'$, the calculation would involve integration in $\thetaspace$ which is computationally prohibitive for large $d$. We show that when the loss function $\ell(\cdot)$ is $\lambda$-strongly convex and $\beta$-smooth, the $I_\q/E_\q$ term in <a href="#renyi_loss_rate">Lemma 2</a> can be controlled along the coupled tracing diffusions. More specifically, by choosing a step size $\step \leq \frac{1}{\beta}$ and start parameter distribution $\thet_0, \thet_0' \sim \proj{\C}{\mathcal{N} (0, \sig^2 I_d)}$, we show that for any $t \geq 0$,
{: .text-justify}

$$
 \frac{\Gren{\q}{\thet_t}{\thet_t'}}{\Fren{\q}{\thet_t}{\thet_t'}} \geq \frac{\lambda} {\q^2\sigma^2}\times \left[\Ren{\q}{\thet_t}{\thet_t'} + \q(\q - 1) \doh{\Ren{\q}{\thet_t}{\thet_t'}}{\q}\right].
$$

We prove this inequality by showing that under the stated conditions, the coupled tracing diffusions satisfies the Gaussian isoperimetric inequality, also known as _log-Sobolev inequality_<d-cite key="bakry2013analysis"></d-cite> (LSI), with constant $\frac{\lambda}{2\sigma^2}$ for all $t\geq 0$. Vempala et al.<d-cite key="vempala2019rapid"></d-cite> proved that the log-Sobolev inequality can be equivalently stated as the bound above on $I_\q/E_\q$.
{: .text-justify}

On plugging the above inequality in <a href="#renyi_loss_rate">Lemma 2</a>, we get the following partial differential inequation (PDI) that models the **dynamics of Rényi privacy loss**, i.e. it describes how the Rényi privacy loss changes as a function of time $t$ and $\alpha$ during the interval $\eta k < t < \eta (k + 1)$. For brevity, let $R(\q,t)$ represent $\Ren{\q}{\thet_t}{\thet_t'}$. We have
{: .text-justify}

$$
\doh{R(\q,t)}{t} \leq \frac{1}{\gamma}\frac{\q S_g^2}{4\sig^2\size^2} - \lambda(1-\gamma) \left[ \frac{R(\q,t)}{\q} + (\q-1)\doh{R(\q,t)}{\q} \right].
$$

As per our <a href="#coupled_diffusion">coupled tracing diffusion</a> process, no privacy loss is incurred at integral multiples of $\step$ as both the mapping $\phi(\cdot)$ and $\proj{\C}{\cdot}$ are identical for the coupled processes, and therefore can be seen as post-processing step (as shown below).
{: .text-justify}

<figure id="privacy_loss_one_step">
<picture>
    <source srcset="{{ site.baseurl }}/assets/img/blog/DPDynamics/privacy_loss_one_step.png" media="(min-width: 768px)" />
    <img srcset="privacy_loss_one_step.png" alt="RDP" />
</picture>
</figure>

On solving the DP dynamics PDI and unrolling it over all the $K$ steps of iteration yields the privacy guarantee stated in <a href="#main_theorem">Theorem 1</a>. <a href="#privacy_loss_total">Figure below</a> demonstrates how this RDP guarantee for noisy GD converges with the number of iterations $\K$. Through y-axis, we show the $\eps$ guaranteed for noisy GD under a reasonable choice of hyperparameters. The RDP order $\q$ linearly scales the asymptotic guarantee, but does not affect the convergence rate of RDP guarantee. However, the strong convexity parameter $\lambda$ positively affects the asymptotic guarantee as well as the convergence rate; the larger the strong convexity parameter $\lambda$ is, the stronger the asymptotic RDP guarantee and the faster the convergence.
{: .text-justify}

<figure id="privacy_loss_total">
<picture>
    <source srcset="{{ site.baseurl }}/assets/img/blog/DPDynamics/privacy_loss_total.png" media="(min-width: 768px)" />
    <img srcset="privacy_loss_total.png" alt="RDP" />
</picture>
</figure>

## Tightness Analysis

Differential privacy guarantees reflect a _bound_ on privacy loss on an algorithm; thus, it is very crucial to also have an analysis of their tightness (i.e., how close they are to the exact privacy loss). We prove that our guarantee in <a href="#main_theorem">Theorem 1</a> is tight. To this end, we construct an instance of the ERM optimization problem, for which we show that the Rényi privacy loss of the noisy GD algorithm grows at an order matching our guarantee in <a href="#main_theorem">Theorem 1</a>.
{: .text-justify}

It is very challenging to lower bound the exact Rényi privacy loss ${\Ren{\q}{\thet_{\K\step}}{\thet_{\K\step}'}}$ in general. This might require having an explicit expression for the probability distribution over the last-iterate parameters $\ptheta_\kk$. Computing a close-form expression is, however, feasible when the loss gradients are linear. This is due to the fact that, after a sequence of linear transformations and Gaussian noise additions, the parameters follow a Gaussian distribution. Therefore, we construct such an ERM objective, compute the exact privacy loss, and prove the following lower bound.
{: .text-justify}

<div id="lower_bound" class="theorem"><b>2.</b>
There exist two neighboring datasets ${\D, \D' \in \Domain}$, a start parameter $\ptheta_0$, and a smooth loss function $\loss{\x}{\ptheta}$ on unconstrained convex set $\C=\mathbb{R}^d$, with a finite total gradient sensitivity $\sen{g}$, such that for any step-size $\step<1$, noise variance $\sig^2>0$, and iteration $\K\in\mathbb{N}$, the privacy loss of $\mathcal{A}_{\text{Noisy-GD}}$ on $\D, \D'$ is lower-bounded by
$$
\Ren{\q}{\thet_{\step\K}}{\thet_{\step\K}'} \geq \frac{\q S_g^2}{4\sig^2 n^2} \left( 1 - e^{-\step\K} \right).
$$
</div>

We prove this lower bound using the $\ell_2$-squared norm loss as ERM objective: $\underset{\theta\in\mathbb{R}^d}{\min}\sum_{i=1}^n\frac{\lVert\theta-\x_i\rVert_2^2}{n}$. We assume bounded data domain s.t. the gradient has finite sensitivity. With start parameter $\theta_0 = 0^d$, the $k^{\text{th}}$ step parameter $\theta_k$ is distributed as Gaussian with mean ${\mu_k=\step\bar{\x}\sum_{i=0}^{k-1}(1-\step)^i}$ and variance ${\sigma_k^2=\frac{2\eta\sigma^2}{\size^2}\sum_{i=0}^{k-1}(1-\step)^{2i}}$ in each dimension, where $\bar{x}=\sum_{i=1}^n\x_i/n$ is the empirical dataset mean. We explicitly compute the privacy loss at any step $\K$, which is lower bounded by ${\frac{\q S_g^2}{4\sig^2n^2}(1-e^{-\eta\K})}$.
{: .text-justify}

Our <a href="#main_theorem">RDP guarantee</a> converges fast to $\frac{\alpha S_g^2}{\sig^2n^2}$, which matches the lower bound at every step $\K$, up to a constant of $4$. This immediately shows tightness of our converging RDP guarantee _throughout_ the training process, for a converging noisy GD algorithm.
{: .text-justify}

## Utility Analysis

The randomness, required for satisfying differential privacy, can adversely affect the utility of the trained model. The standard way to measure the utility of a randomized ERM algorithm (for example, $\mathcal{A}\_{\text{Noisy-GD}}$) is to quantify its worst case _excess empirical risk_, which is defined as
{: .text-justify}

$$
\max_{\D \in \Domain}\Expec{}{\Loss_{\D}(\ptheta) - \Loss_{\D}(\ptheta^*)},
$$

where $\ptheta$ is the output of the randomized algorithm $\mathcal{A}\_{\text{Noisy-GD}}$ on $\D$, $\ptheta^*$ is the optimal solution to the standard (no privacy) ERM on $\Loss_\D$, and the expectation is computed over the randomness of the algorithm.
{: .text-justify}

We provide the _optimal_ excess empirical risk (utility) of noisy GD algorithm under an $(\eps, \delta)$-DP constraint. The notion of _optimality_ for utility is defined as the smallest upper-bound for excess empirical risk that can be guaranteed under $(\eps, \delta)$-DP constraint by tuning the algorithm's hyperparameters (such as the noise variance $\sig^2$ and the number of iterations $\K$).  
{: .text-justify}

<div class="theorem" id="optimal_empirical_risk"><b>3 (Empirical risk upper bound for $(\eps,\delta)$-DP Noisy GD).</b> 
For Lipschitz smooth strongly convex loss function $\ell(\theta;\x)$ on a bounded closed convex set $\mathcal{C}\subseteq\thetaspace$, and dataset $\D\in\Domain$ of size $n$, if the step-size $\eta=\frac{\lambda}{2\beta^2}$ and the initial parameter $\theta_0\sim\proj{\C}{\mathcal{N}(0,\frac{2\sig^2}{\lambda} I_d)}$, then the <a href="#goal-noisygd">noisy GD Algorithm</a> is $(\eps,\delta)$-differentially private, and satisfies
$$
	\mathbb{E}[\Loss_D(\theta_{K^*})-\Loss_D(\theta^*)]=O(\frac{\beta d L^2\log(1/\delta)}{\eps^2\lambda^2\size^2}),
$$
by setting noise variance $\sig^2=\frac{8L^2 (\eps+2\log(1/\delta))}{\lambda\eps^2 n^2}$, and number of updates $K^*=\frac{2\beta^2}{\lambda^2}\log(\frac{n^2\eps^2}{4\log(1/\delta) d})$.
</div>

This utility matches the following theoretical lower bound in Bassily et al.<d-cite key="bassily2014private"> for the best attainable utility of $(\eps,\delta)$-differentially private algorithms on Lipschitz smooth strongly convex loss functions.

<div class="theorem" id="optimal_empirical_risk"><b>4 (<d-cite key="bassily2014private"></d-cite>Empirical risk lower bound for $(\eps,\delta)$-DP).</b> 
Let $\size, d \in \mathbb N$, $\eps > 0$, and $\delta=o(\frac{1}{n})$. For every $(\eps, \delta)$-differentially private algorithm $\mathcal A$ (whose output is denoted by $\theta^{priv}$), then
$$
\mathbb{E}[\Loss_\D(\ptheta^{priv}) - \Loss_\D(\ptheta^*)] = \Omega\left(\min\left\{1, \frac{d}{\eps^2 \size^2}\right\}\right),
$$
where $\ptheta^*$ minimizes a constructed $1$-Lipschitz, $1$-strongly convex objective $\mathcal{L}_D(\theta)$ over convex set $\C$.
</div>

<figure id="utility_table">
<picture>
    <source srcset="{{ site.baseurl }}/assets/img/blog/DPDynamics/utility_table.png" media="(min-width: 768px)" />
    <img srcset="utility_table.png" alt="RDP" />
</picture>
</figure>

Our utility matches this lower bound upto the constant factor $\log(1/\delta)$, when assuming $\frac{\beta}{\lambda^2}=O(1)$. This improves upon the previous gradient perturbation methods<d-cite key="bassily2014private,wang2018differentially"></d-cite> by a factor of $\log(n)$, and matches the utility of previously know optimal ERM algorithm for Lipschitz smooth strongly convex loss functions, such as objective perturbation<d-cite key="chaudhuri2011differentially,kifer2012private"></d-cite> and output perturbation<d-cite key="zhang2017efficient"></d-cite>. As shown in <a href="#utility_table">Table 1</a>, our utility guarantee for noisy GD is logarithmically better than that for noisy SGD in Bassily et al.<d-cite key="bassily2014private"></d-cite>, although the two algorithms are extremely similar. This is because we use our tight RDP guarantee, while Bassily et al.<d-cite key="bassily2014private"></d-cite> use a composition-based privacy bound. More specifically, noisy SGD needs $n^2$ iterations to achieve the optimal utility, as shown in <a href="#utility_table">Table 1</a>. This number of iterations is large enough for the composition-based privacy bound to grow above our RDP guarantee, thus leaving room for improving privacy utility trade-off. This concludes that our tight privacy guarantee enables providing a superior privacy-utility trade-off, for Lipschitz, strongly convex, and smooth loss functions.
{: .text-justify}

Our algorithm also has significantly smaller gradient complexity than noisy SGD<d-cite key="bassily2014private"></d-cite>, for strongly convex loss functions, by a factor of ${n}/{\log n}$. We use a (moderately large) constant step-size, thus achieving fast convergence to optimal utility. However, noisy SGD <d-cite key="bassily2014private"></d-cite> uses a decreasing step-size, thus requiring more iterations to reach optimal utility.
{: .text-justify}

## Conclusion

We have developed a novel theoretical framework for analyzing the dynamics of privacy loss for noisy gradient descent algorithms. Our theoretical results show that by hiding the internal state of the training algorithm (over many iterations over the whole data), we can tightly analyze the rate of information leakage throughout training, and derive a bound that is significantly tighter than that of composition-based approaches.
{: .text-justify}

**Future Work.** Our main result is a tight privacy guarantee for Noisy GD on smooth and strongly convex loss functions. The assumptions are very similar to that of the prior work on privacy amplification by iteration<d-cite key="feldman2018privacy"></d-cite>, and have obvious advantages in enabling the tightness and utility analysis. However, the remaining open challenge is to extend this analysis to non-smooth and non-convex loss functions, and stochastic gradient updates, which are used notably in deep learning.
{: .text-justify}
