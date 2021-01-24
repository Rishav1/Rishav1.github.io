---
layout: post
title:  "Understanding Generative Adverserial Imitation Learning"
date:   2018-11-17
desc: "Understanding Generative Adverserial Imitation Learning"
keywords: "Reinforcement Learning, Inverse Imitation Learning"
categories: [Research]
tags: [Reinforcement Learning, Inverse Imitation Learning]
icon: icon-html
---

Imitation Learning, as the name suggests, is the theory of teaching an agent mimic to an expert. In typical Reinforcement Learning MDP setting, this task equates to mimicing behaviour of an expert policy, which can not be accessed directly(i.e. you cannot see/generate the policy table). The only way to get the knowledge about the policy is via a sample of expert's trajectories, i.e. traces of the expert policy on the MDP(environment) in question. This task seems simple, your imitation policy should just remeber what the expert did, correct? Not quite so. The problem is that you have only a limited set of expert trajectories(say 100 runs). What if you encounter a state that the expert never visited?
{: .text-justify}

<p align="center"><img src="https://assets.rbl.ms/14058095/980x.jpg
" alt="blog-image" width="480" height="320"></p>
<p align="center">Behaviour Cloning XD.</p>

There are quite a few approaches to solving this problem of Imitation Learning. My favourite one so far that I am going to explain in this blog is the [Generative Adverserial Imitation Learning](https://arxiv.org/abs/1606.03476). But to be able to appreciate it, lets go over some other approaches that lead upto this. First, let's get over some notations.
{: .text-justify}


The environment is asumed to be a typical MDP with state space ![state] and action space ![action], a transition probability ![transition] and a cost function ![cost]. The slight difference with RL here is that we don't actually know the cost function. However we do have some sampled expert trajectories ![trajectory], which we can use to forumlate a cost function. Here, ![pi] is the expert policy.
{: .text-justify}

__Behavorial Cloning:__ The most basic idea to do imitation learning is to view it as supervised learning problem. You train a predictor to imitate the expert by training it on ![sample]. This is similar to remebering the agent policy as this method dosen't generalize to most states that weren't seen, though actions at some unseen similar states would be very similar to the expert. The problem is that once the agent deviates from the expert at some point and observes a state not in any expert trajectory it trained on, the agent's action becomes very different from what the expert would have taken, and this divergence only grows with time from that point on.
{: .text-justify}

<p align="center"><img src="https://imgur.com/6PhNetM.png" alt="blog-image" width="480" height="320"></p>
<p align="center">Behaviour Cloning XD.</p>

__Inverse Reinfocement Learning:__ One idea to imitate the expert is to try to find the cost function that has low costs(say 0) on action choices seen in expert trajectories and high costs(say 1) on all other trajectories. Based on this cost function, you could then train an agent, which should learn to imitate the expert. This is an improvement over Behavorial cloning in the sense that you could, in theory, do infinite simulations which would teach the agent how to quickly recover from a diverged path. This isn't however a practical solution, as very close but not identical agent trajectories will have the same outrageously high cost as any other radically different trajectory.
{: .text-justify}

What we ideally want is a cost function that gives low scores for policies generating trajectories quite similar to the ones generated using expert policy, but high for others. Moreover, the cost should be sufficiently smooth and non-sparse for the RL to be able to learn properly. What cost function would do? We can sit down an engineer a static cost functions for this task, but that is no fun. Who will configure the parameters for every imitation learning problem? There is a pretty neat solution to this. Why not use a cost function that is based on the current agent policy? Who said the cost function has to be static? Take a look at the following evolving cost functions used in Apprenticeship Learning.
{: .text-justify}

<p align="center"><img src="https://latex.codecogs.com/gif.latex?c_t=\max_{c\in&space;C_{selective}}&space;E_{\pi_t}[c(s,a))]&space;-&space;E_{\pi_e}[c(s,a)]" alt="blog-image"></p>

__Apprenticeship Learning:__ The above cost function is dynamic in terms of the learnt policy. It selects a cost function that best penealizes a policy based on how far it is from the expert. The catch is that there is a constraint on the cost functions that can be selected. This is to ensure that a very difficult to learn cost functions isn't selected(such as the 0,1-static policy mentioned earlier). Plus one can carefully engineer a class of cost functions that is tractable to compute as well as allows fast learning. [This seminal idea](https://ai.stanford.edu/~ang/papers/icml04-apprentice.pdf), called Apprenticeship Learning, was proposed by [Andrew Ng](https://twitter.com/AndrewYNg) and [Pieter Abbeel](https://twitter.com/pabbeel) back in 2004. And was a significant improvement over previous IL methods.
{: .text-justify}

Apprenticeship Learning also presented a cunning and scalable way of doing IL in very large state-action spaces, which in my opinion was the grand prize. In brief, what they did was approximate the policy function and parameterize the cost function via linear weights on a set of selected feature cost functions. So the space of cost functions ![cost_f] became a convex set of a preselected cost functions ![phi]. So any cost function could be represented using a weight __W__ as
{: .text-justify}

<p align="center"><img src="https://latex.codecogs.com/gif.latex?c_{W}(s,a)&space;=&space;W^T\phi(s,a),&space;&space;W&space;\in&space;[0,1]^k." alt="blog-image"></p>

Given any set of trajectories(say m in number)

<p align="center"><img src="https://latex.codecogs.com/gif.latex?\tau=\{\{s_0^{(0)},a_0^{(0)},s_1^{(0)},...\},\{s^{(1)},a^{(1)},s_1^{(1)},...}\},...,\{s_0^{(m)},a_0^{(m)},s_1^{(m)},...\}\}," alt="blog-image"></p>
one could estimate the feature cost expectation for the set of trajectories using the following

<p align="center"><img src="https://latex.codecogs.com/gif.latex?\mu(\tau)&space;=&space;\frac{1}{m}\sum_{i=1}^m\sum_{t=0}^\infty&space;\gamma&space;\phi(s_t^{(i)},a_t^{(i)})." alt="blog-image"></p>

The algorithm then selected a weight __W__ that gave the maximum differene between the agent's trajectories and the expert's trajectories. So basically, the cost function keeps changing to ensure that the penalty between the agent's trajectories and expert trajectories remains as high as possible given the constraints on the feature cost functions.


<div>
  <style type='text/css'>
    @import url('https://fonts.googleapis.com/css?family=Lato:300,400,700');
    /* list styles */
    span {
      line-height: 25px;
    }
    dafsda {
      font-family: 'Lato', sans-serif;
      padding: 10px 20px;
      max-width: 500px;
      width: 98%;
      margin-left: auto;
      margin-right: auto;
      box-sizing: border-box;
    }
    h1 {
      color: #373a3c;
      font-size: 38px;
    	margin: 0 0 12px 0;
    	line-height: 1em;
      font-weight: 300;
    }
    h2 {
    	font-size: 32px;
    	margin: 0 0 12px 0;
    	line-height: 1em;
      font-weight: 300;
      color: #373a3c;
    }
    h3 {
    	font-size: 25px;
    	margin: 0 0 12px 0;
    	line-height: 1em;
      font-weight: 300;
      color: #373a3c;
    }
    h4 {
    	font-size: 18.75px;
    	line-height: 1em;
    	margin: 0 0 12px 0;
      font-weight: 300;
      color: #373a3c;
    }
    h5 {
    	font-size: 16px;
    	margin: 0 0 12px 0;
    	line-height: 1em;
      font-weight: 300;
      color: #373a3c;
    }
    p {
    	font-size: 15px;
    	margin: 0 0 12px 0;
    	line-height: 1.4em;
      color: #373a3c;
    }
    li {
      font-size: 15px;
      line-height: 1.4em;
      color: #373a3c;
    }
    ul, ol {
      margin: 12px 0 12px 2em;
      padding: 0;
    }
    a:link, a:visited {
      color: cornflowerblue;
    }
    a:hover, a:active {
      color: midnightblue;
    }  
  </style>
  <div id="dafsda">
  <h3>The Apprenticeship Learning algorithm.</h3>
<ol style="list-style: none;">
  <li style="margin: 20px 0;"><span style="font-weight: 700; font-size: 150%; position: absolute;">01</span><p style="padding-left: 40px; position: relative;">Given expert trajectory <img src="https://latex.codecogs.com/gif.latex?\tau_e" alt="blog-image">, estimate the preselected cost functions <img src="https://latex.codecogs.com/gif.latex?\mu_e&space;=&space;\mu(\tau_e)" alt="blog-image">.</p></li>
  <li style="margin: 20px 0;"><span style="font-weight: 700; font-size: 150%; position: absolute;">02</span><p style="padding-left: 40px; position: relative;">Set i = 0. Randomly pick a policy <img src="https://latex.codecogs.com/gif.latex?\pi_i" alt="blog-image">.</p></li>
  <li style="margin: 20px 0;"><span style="font-weight: 700; font-size: 150%; position: absolute;">03</span><p style="padding-left: 40px; position: relative;">Generate a few trajectories <img src="https://latex.codecogs.com/gif.latex?\tau_i" alt="blog-image"> using policy <img src="https://latex.codecogs.com/gif.latex?\pi_i" alt="blog-image">.</p></li>
  <li style="margin: 20px 0;"><span style="font-weight: 700; font-size: 150%; position: absolute;">04</span><p style="padding-left: 40px; position: relative;">Estimate <img src="https://latex.codecogs.com/gif.latex?\mu_i&space;=&space;\mu(\tau_i)" alg="blog-image">.</p></li>
  <li style="margin: 20px 0;"><span style="font-weight: 700; font-size: 150%; position: absolute;">05</span><p style="padding-left: 40px; position: relative;">Compute <img src="https://latex.codecogs.com/gif.latex?W_i&space;=&space;arg\max_{W\in&space;[0,1]^k}&space;W^T(\mu_i&space;-&space;\mu_e)" alt="blog-image">. Halt if |W<sub>i</sub>| is less than ϵ.</p></li>
  <li style="margin: 20px 0;"><span style="font-weight: 700; font-size: 150%; position: absolute;">06</span><p style="padding-left: 40px; position: relative;">Perform RL using the cost function <img src="https://latex.codecogs.com/gif.latex?c(s,a)=W_i^T\phi(s,a)">, generating trajectories <img src="https://latex.codecogs.com/gif.latex?\tau_{i+1}">.</p></li>
  <li style="margin: 20px 0;"><span style="font-weight: 700; font-size: 150%; position: absolute;">07</span><p style="padding-left: 40px; position: relative;">Set i = i+1. Go to step 4.</p></li>
</ol>
</div>
</div>

What Apperentice Learning algorithm does can be considered to be a very basic form of GAIL. Notice that there is a continuous cycle of __RL__ and __W__ update? This alternating updates is opossing in nature; while RL decreases the cost incurred by agent, feature weight update increases it. This alternating behavior actually is what drives IL.

__Generative Adverserial Imitation Learning:__ GAIL paper sheds more light into this idea of doing Inverse Reinforcement Learning and Reinforcement Learning in an iterated sequence. It generalizes the idea of Apprenticeship Learning by introducing a discriminator network __D<sub>w</sub>__, parameterized by __w__, that roughly does the task of the cost function, i.e. penealizing policies different from the expert. Using ideas from [Generative Adverserial Networks](https://papers.nips.cc/paper/5423-generative-adversarial-nets.pdf), this discriminator network __D<sub>w</sub>__ can be trained to be able to learn to differenciate better and better using the following gradient.

<p align="center"><img src="https://latex.codecogs.com/gif.latex?\hat{\mathbb{E}}_{\tau_i}[\nabla_w&space;log(D_w(s,a))]&space;&plus;&space;\hat{\mathbb{E}}_{\tau_e}[\nabla_w&space;log(1&space;-D_w(s,a))]" alt="blog-image"></p>

This is analogus to updating the feature costs' weights in Apprenticeship Learning. Rest of the idea remains the same roughly. In GAIL, the authors have used Trust Region Policy Optimization as the RL algorithm, because according to them it improves the imitation, but in my opinion any suitable policy-parameterized RL algorithm could be used, with the cost function as __c(s,a) = log(D<sub>w</sub>(s,a))__.

The GAIL paper seems cool to me not because it proposed a neat dynamic cost function that leads to imitation, but because they showed that finding the optimal imitation policy through RL and finding the best discrimintation cost function through IRL are two faces of the same coin, i.e. they are the Min-Max and the Max-Min solutions of the same following bivariate function.

<p align="center"><img src="https://latex.codecogs.com/gif.latex?\bar{L}(\rho,&space;c)&space;=&space;-\bar{H}(\rho)&space;-&space;\psi(c)&space;&plus;&space;\sum_{s,a}\rho(s,a)c(s,a)&space;-&space;\sum_{s,a}\rho_{\pi_E}(s,a)c(s,a)" alt="blog-image"></p>

The occupancy measure ![oco] of optimal policy ![pio] and the optimal discriminator cost function ![co] can be expressed in terms of the above bivariate function as following.

<p align="center"><img src="https://latex.codecogs.com/gif.latex?\rho^*&space;\in&space;\arg&space;\min_{\rho}&space;\max_c&space;\bar{L}(\rho,&space;c),&space;\quad&space;c^*&space;\in&space;\arg&space;\max_{c}&space;\min_{\rho}&space;\bar{L}(\rho,&space;c)" alt="blog-image"></p>

Using the MinMax theorem, the authors established that if the residual part of __L__ is strictly convex, then the solution is unique and can be represented as the conjugate of the residual, which to me is pretty awesome thing to prove.









[pio]:https://latex.codecogs.com/gif.latex?\pi^*
[oco]:https://latex.codecogs.com/gif.latex?\rho^*
[co]:https://latex.codecogs.com/gif.latex?c^*

[pi]:https://latex.codecogs.com/gif.latex?\pi_e&space;|&space;S&space;\rightarrow&space;A
[state]:https://latex.codecogs.com/gif.latex?S
[action]:https://latex.codecogs.com/gif.latex?A
[transition]:https://latex.codecogs.com/gif.latex?P(s'|s,a)
[cost]:https://latex.codecogs.com/gif.latex?c\in&space;C&space;=&space;\{c|&space;S\times&space;A&space;\rightarrow&space;R\}
[trajectory]:https://latex.codecogs.com/gif.latex?\tau_e&space;\sim&space;\pi_e
[sample]:https://latex.codecogs.com/gif.latex?\{(s_1,a_1),(s_2,a_2),...,(s_n,a_n)&space;\}&space;\sim&space;\tau_e
[cost_function]:https://latex.codecogs.com/gif.latex?\max_{c\in&space;C}&space;E_\pi[c(s,a))]&space;-&space;E_\pi_e[c(s,a)]
[cost_f]:https://latex.codecogs.com/gif.latex?C_{selective}
[phi]:https://latex.codecogs.com/gif.latex?\phi&space;=&space;\{c_1,&space;c_2,&space;...,&space;c_k\}
[c]:https://latex.codecogs.com/gif.latex?c_{W}(s,a)&space;=&space;W^T\phi(s,a),&space;&space;W&space;\in&space;[0,1]^k
[tau_e]:https://latex.codecogs.com/gif.latex?\tau_e
[tau_j]:https://latex.codecogs.com/gif.latex?\tau_{i+1}
[tau_i]:https://latex.codecogs.com/gif.latex?\tau_{i}
[mu_e]:https://latex.codecogs.com/gif.latex?\mu_e&space;=&space;\mu(\tau_e)
[mu_i]:https://latex.codecogs.com/gif.latex?\mu_e&space;=&space;\mu(\tau_i)
[mu_j]:https://latex.codecogs.com/gif.latex?\mu_e&space;=&space;\mu(\tau_{i+1})
[tau]:https://latex.codecogs.com/gif.latex?\tau=\{\{s_0^{(0)},a_0^{(0)},s_1^{(0)},...\},\{s^{(1)},a^{(1)},s_1^{(1)},...}\},...,\{s_0^{(m)},a_0^{(m)},s_1^{(m)},...\}\}
[mu]:https://latex.codecogs.com/gif.latex?\mu(\tau)&space;=&space;\frac{1}{m}\sum_{i=1}^m\sum_{t=0}^\infty&space;\gamma&space;\phi(s_t^{(i)},a_t^{(i)}))))
[pi_i]:https://latex.codecogs.com/gif.latex?\pi_i
[maxW]:https://latex.codecogs.com/gif.latex?W_i&space;=&space;arg\max_{W\in&space;[0,1]^k}&space;W^T(\mu_i&space;-&space;\mu_e)
[c_fn]:https://latex.codecogs.com/gif.latex?c(s,a)=W_i^T\phi(s,a)
