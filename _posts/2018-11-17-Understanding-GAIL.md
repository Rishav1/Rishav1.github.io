---
layout: post
title: "Understanding Generative Adverserial Imitation Learning"
date: 2018-11-17
description: "Understanding Generative Adverserial Imitation Learning"
keywords: "Reinforcement Learning, Inverse Imitation Learning"
categories: [Research]
tags: [Reinforcement Learning, Inverse Imitation Learning]
icon: icon-html
comments: true
mathjax: true
---

Imitation Learning, as the name suggests, is the theory of teaching an agent mimic to an expert. In typical Reinforcement Learning MDP setting, this task equates to mimicing behaviour of an expert policy, which can not be accessed directly(i.e. you cannot see/generate the policy table). The only way to get the knowledge about the policy is via a sample of expert's trajectories, i.e. traces of the expert policy on the MDP(environment) in question. This task seems simple, your imitation policy should just remeber what the expert did, correct? Not quite so. The problem is that you have only a limited set of expert trajectories(say 100 runs). What if you encounter a state that the expert never visited?
{: .text-justify}

<p align="center"><img src="{{ '/assets/img/blog/GAIL/behaviour-cloning-1.jpg' | relative_url }}" alt="Behaviour cloning" width="480"></p>
<p align="center">Behaviour Cloning XD.</p>

There are quite a few approaches to solving this problem of Imitation Learning. My favourite one so far that I am going to explain in this blog is the [Generative Adverserial Imitation Learning](https://arxiv.org/abs/1606.03476). But to be able to appreciate it, lets go over some other approaches that lead upto this. First, let's get over some notations.
{: .text-justify}

The environment is asumed to be a typical MDP with state space $S$ and action space $A$, a transition probability $P(s'|s,a)$ and a cost function $c \in C = \{c \mid S \times A \rightarrow \mathbb{R}\}$. The slight difference with RL here is that we don't actually know the cost function. However we do have some sampled expert trajectories $\tau_e \sim \pi_e$, which we can use to forumlate a cost function. Here, $\pi_e : S \rightarrow A$ is the expert policy.
{: .text-justify}

**Behavorial Cloning:** The most basic idea to do imitation learning is to view it as supervised learning problem. You train a predictor to imitate the expert by training it on $\{(s_1,a_1),(s_2,a_2),\dots,(s_n,a_n)\} \sim \tau_e$. This is similar to remebering the agent policy as this method dosen't generalize to most states that weren't seen, though actions at some unseen similar states would be very similar to the expert. The problem is that once the agent deviates from the expert at some point and observes a state not in any expert trajectory it trained on, the agent's action becomes very different from what the expert would have taken, and this divergence only grows with time from that point on.
{: .text-justify}

<p align="center"><img src="{{ '/assets/img/blog/GAIL/behaviour-cloning-2.jpg' | relative_url }}" alt="Behaviour cloning" width="480"></p>
<p align="center">Compounding errors in behaviour cloning.</p>

**Inverse Reinfocement Learning:** One idea to imitate the expert is to try to find the cost function that has low costs(say 0) on action choices seen in expert trajectories and high costs(say 1) on all other trajectories. Based on this cost function, you could then train an agent, which should learn to imitate the expert. This is an improvement over Behavorial cloning in the sense that you could, in theory, do infinite simulations which would teach the agent how to quickly recover from a diverged path. This isn't however a practical solution, as very close but not identical agent trajectories will have the same outrageously high cost as any other radically different trajectory.
{: .text-justify}

What we ideally want is a cost function that gives low scores for policies generating trajectories quite similar to the ones generated using expert policy, but high for others. Moreover, the cost should be sufficiently smooth and non-sparse for the RL to be able to learn properly. What cost function would do? We can sit down an engineer a static cost functions for this task, but that is no fun. Who will configure the parameters for every imitation learning problem? There is a pretty neat solution to this. Why not use a cost function that is based on the current agent policy? Who said the cost function has to be static? Take a look at the following evolving cost functions used in Apprenticeship Learning.
{: .text-justify}

$$
c_t = \max_{c \in C_{selective}} E_{\pi_t}[c(s,a)] - E_{\pi_e}[c(s,a)]
$$

**Apprenticeship Learning:** The above cost function is dynamic in terms of the learnt policy. It selects a cost function that best penealizes a policy based on how far it is from the expert. The catch is that there is a constraint on the cost functions that can be selected. This is to ensure that a very difficult to learn cost functions isn't selected(such as the 0,1-static policy mentioned earlier). Plus one can carefully engineer a class of cost functions that is tractable to compute as well as allows fast learning. [This seminal idea](https://ai.stanford.edu/~ang/papers/icml04-apprentice.pdf), called Apprenticeship Learning, was proposed by [Andrew Ng](https://twitter.com/AndrewYNg) and [Pieter Abbeel](https://twitter.com/pabbeel) back in 2004. And was a significant improvement over previous IL methods.
{: .text-justify}

Apprenticeship Learning also presented a cunning and scalable way of doing IL in very large state-action spaces, which in my opinion was the grand prize. In brief, what they did was approximate the policy function and parameterize the cost function via linear weights on a set of selected feature cost functions. So the space of cost functions $C_{selective}$ became a convex set of a preselected cost functions $\phi = \{c_1, c_2, \dots, c_k\}$. So any cost function could be represented using a weight **W** as
{: .text-justify}

$$
c_{W}(s,a) = W^T \phi(s,a), \quad W \in [0,1]^k.
$$

Given any set of trajectories(say m in number)

$$
\tau = \{\{s_0^{(0)},a_0^{(0)},s_1^{(0)},\dots\}, \{s^{(1)},a^{(1)},s_1^{(1)},\dots\}, \dots, \{s_0^{(m)},a_0^{(m)},s_1^{(m)},\dots\}\},
$$

one could estimate the feature cost expectation for the set of trajectories using the following

$$
\mu(\tau) = \frac{1}{m} \sum_{i=1}^m \sum_{t=0}^\infty \gamma\, \phi(s_t^{(i)}, a_t^{(i)}).
$$

The algorithm then selected a weight **W** that gave the maximum differene between the agent's trajectories and the expert's trajectories. So basically, the cost function keeps changing to ensure that the penalty between the agent's trajectories and expert trajectories remains as high as possible given the constraints on the feature cost functions.

<div class="algorithm-box">
  <h4>The Apprenticeship Learning algorithm.</h4>
  <ol>
    <li>Given expert trajectory $\tau_e$, estimate the preselected cost functions $\mu_e = \mu(\tau_e)$.</li>
    <li>Set $i = 0$. Randomly pick a policy $\pi_i$.</li>
    <li>Generate a few trajectories $\tau_i$ using policy $\pi_i$.</li>
    <li>Estimate $\mu_i = \mu(\tau_i)$.</li>
    <li>Compute $W_i = \arg\max_{W \in [0,1]^k} W^T(\mu_i - \mu_e)$. Halt if $|W_i|$ is less than $\epsilon$.</li>
    <li>Perform RL using the cost function $c(s,a) = W_i^T \phi(s,a)$, generating trajectories $\tau_{i+1}$.</li>
    <li>Set $i = i+1$. Go to step 4.</li>
  </ol>
</div>

What Apperentice Learning algorithm does can be considered to be a very basic form of GAIL. Notice that there is a continuous cycle of **RL** and **W** update? This alternating updates is opossing in nature; while RL decreases the cost incurred by agent, feature weight update increases it. This alternating behavior actually is what drives IL.

**Generative Adverserial Imitation Learning:** GAIL paper sheds more light into this idea of doing Inverse Reinforcement Learning and Reinforcement Learning in an iterated sequence. It generalizes the idea of Apprenticeship Learning by introducing a discriminator network **D<sub>w</sub>**, parameterized by **w**, that roughly does the task of the cost function, i.e. penealizing policies different from the expert. Using ideas from [Generative Adverserial Networks](https://papers.nips.cc/paper/5423-generative-adversarial-nets.pdf), this discriminator network **D<sub>w</sub>** can be trained to be able to learn to differenciate better and better using the following gradient.

$$
\hat{\mathbb{E}}_{\tau_i}[\nabla_w \log(D_w(s,a))] + \hat{\mathbb{E}}_{\tau_e}[\nabla_w \log(1 - D_w(s,a))]
$$

This is analogus to updating the feature costs' weights in Apprenticeship Learning. Rest of the idea remains the same roughly. In GAIL, the authors have used Trust Region Policy Optimization as the RL algorithm, because according to them it improves the imitation, but in my opinion any suitable policy-parameterized RL algorithm could be used, with the cost function as **c(s,a) = log(D<sub>w</sub>(s,a))**.

The GAIL paper seems cool to me not because it proposed a neat dynamic cost function that leads to imitation, but because they showed that finding the optimal imitation policy through RL and finding the best discrimintation cost function through IRL are two faces of the same coin, i.e. they are the Min-Max and the Max-Min solutions of the same following bivariate function.

$$
\bar{L}(\rho, c) = -\bar{H}(\rho) - \psi(c) + \sum_{s,a} \rho(s,a) c(s,a) - \sum_{s,a} \rho_{\pi_E}(s,a) c(s,a)
$$

The occupancy measure $\rho^*$ of optimal policy $\pi^*$ and the optimal discriminator cost function $c^*$ can be expressed in terms of the above bivariate function as following.

$$
\rho^* \in \arg\min_{\rho} \max_c \bar{L}(\rho, c), \quad c^* \in \arg\max_{c} \min_{\rho} \bar{L}(\rho, c)
$$

Using the MinMax theorem, the authors established that if the residual part of **L** is strictly convex, then the solution is unique and can be represented as the conjugate of the residual, which to me is pretty awesome thing to prove.
