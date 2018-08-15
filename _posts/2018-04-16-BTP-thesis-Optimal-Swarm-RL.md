---
layout: post
title:  "My Bachelor's Thesis on Improving Exploration in Ensemble DQN"
date:   2018-04-16
desc: "My Bachelor's Thesis on Improving Exploration in Ensemble Deep Q-Learning Networks"
keywords: "Reinforcement Learning, Ensemble, Q-Learning, DQN"
categories: [Research]
tags: [Reinforcement Learning, Ensemble, Q-Learning, DQN]
icon: icon-html
---

The motivation behind my project was to explore possibility of utilizing knowledge from multiple Reinforcement Learning agents learning to help boost the learning performance. Since sharing of knowledge makes sense only if every agent understands or could benefit from it, the environment for every agent has to be identical(i.e. environment obeys same rules but can be in different states). The idea of combining multiple agent's knowledge is not new. Techniques like Ensemble voting, where a bunch of predictors/agents are trained together and decisions having majority votes are taken, have already been proposed. Another interesting idea is to bootstrap sample agents, i.e. in a set of agents we randomly select one to train for an episode (or fixed time steps), but the data (experience tuples) generated are shared with everyone to learn from. In my thesis below, I proposed a class of algorithms for ensembles, called Swarm RL, that have guarantees on convergence and optimality. Furthermore this class includes both Ensemble Voting and Bootstrap Q-learning within it. The major contribution of the paper is proposal of Optimal Swarm RL, a regret optimal algorithm in this class that theoretically outperforms every other algorithm in the group. The idea of Optimal Swarm RL is to dynamically cluster agents into groups(both number of groups and strength varies for every observation) for making decisions that yield highest coherence. We perform tests on Atari environments(incomplete due to lack of processing power), comparing the proposed algorithm with Bootstrap and Ensemble Voting.
{: .text-justify}

<!-- In my opinion, such techniques are passive in the sense that the group does not directly influence the action choice of it's agents. The group only indirectly affects the learning curve by providing broad learning samples in case of Bootstrap or statistically better samples in case of Ensemble Voting. In order to create methods for active influence on the agents due to the group, I needed to dive into the core of most RL algorithms, i.e. the Bellman Operator and tamper with it. A (very interesting paper)[https://arxiv.org/pdf/1512.04860.pdf] educated me that Bellman operator can be subtly modified in ways that don't affect the optimal convergence yet achieves interesting results.
{: .text-justify} -->


<object data="/research_files/BTP-thesis.pdf" type="application/pdf" style="width:100%" height="700px">
    <embed src="/research_files/BTP-thesis.pdf">
        <p>This browser does not support PDFs. Please download the PDF to view it: <a href="/research_files/BTP-thesis.pdf">Download PDF</a>.</p>
    </embed>
</object>

BibTeX for my Bachelor's Thesis titled __Optimal Swarm RL: An Improved Deep Exploration Strategy__.
```
@misc{Indian Institute of Technology, Guwahati, title={Optimal Swarm RL: An Improved Deep Exploration Strategy}, url={https://rishav1.github.io/research_files/BTP-thesis.pdf}, journal={Rishav Chourasia's Blog}, publisher={Rishav Chourasia}, author={Rishav Chourasia}}
```
