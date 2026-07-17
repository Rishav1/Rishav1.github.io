---
layout: distill
mathjax: true
title: "Failure in Machine Unlearning and How to Guarantee Deletion"
date: 2023-02-15 11:00:00 +0800
categories: [Research]
tags: [Machine Unlearning, Differential Privacy]
bibliography: /2023-02-15-ForgetUnlearning.bib
authors:
  - name: Rishav Chourasia
    url: https://comp.nus.edu.sg/~rishav1
    affiliations:
      name: NUS, Singapore
  - name: Neil Shah
    url: https://www.linkedin.com/in/neil-shah-
toc:
  - name: Machine Unlearning
  - subsections:
      - name: Adaptive Unlearning
  - name: Flaws in Unlearning
  - subsections:
      - name: Threat Model
      - name: Failure under Adaptivity
      - name: Failure with Hidden-States
      - name: Incompleteness
  - name: Trustworthy Data-Deletion
  - subsections:
      - name: Link to Differential Privacy
  - name: Conclusion
---

<p style="display: none;">
$$
\newcommand{\dif}[1]{\mathrm{d} #1}
\newcommand{\der}[2]{\frac{\dif{#1}}{\dif{#2}}}
\newcommand{\doh}[2]{\frac{\partial #1}{\partial #2}}
\newcommand{\graD}[1]{\nabla #1}
\newcommand{\divR}[1]{\mathrm{div} \left(#1\right)}
\newcommand{\lapL}[1]{\Delta #1}
\newcommand{\hesS}[1]{\nabla^2 #1}
\newcommand{\jacO}[1]{\mathbf{J} #1}
\newcommand{\tra}[1]{\textsf{Tr}\left(#1\right)}
\newcommand{\dotP}[2]{\left\langle#1,#2\right\rangle}
\newcommand{\detR}{\mathrm{det}}
\newcommand{\norm}[1]{\left\Vert #1 \right\Vert_2}
\newcommand{\ve}{\ensuremath{\mathbf v}}
\newcommand{\V}{V}

\newcommand{\prob}[2]{\underset{#1}{\mathbb{P}}\left[#2\right]}
\newcommand{\pspace}{\mathcal{P}}
\newcommand{\lap}[1]{\text{Lap}\left(#1\right)}
\newcommand{\expec}[2]{\underset{#1}{\mathbb{E}}\left[#2\right]}
\newcommand{\Gaus}[2]{\mathcal{N}\left(#1, #2\right)}
\newcommand{\Id}{\mathbb{I}\_\dime}
\newcommand{\noise}{\sigma}
\newcommand{\Z}{\mathbf{Z}}
\newcommand{\forget}{u}
\newcommand{\Out}{O}
\newcommand{\Outb}{\bar{\Out}}
\newcommand{\Ze}{\mathbf{W}}
\newcommand{\indic}[1]{\mathbbm{1}\left\{#1\right\}}
\newcommand{\W}{\mathbf{W}}
\newcommand{\pI}{\uppi}
\newcommand{\psI}{\psi}
\newcommand{\Gen}{\mathcal G}
\newcommand{\carre}{\Gamma}
\newcommand{\Ent}{\mathrm{Ent}}
\newcommand{\entropy}{\textrm H}
\newcommand{\p}{p}
\newcommand{\nU}{\upnu}
\newcommand{\mU}{\upmu}
\newcommand{\rhO}{\uprho}

\newcommand{\X}{\mathcal{X}}
\newcommand{\Y}{\mathcal{Y}}
\newcommand{\C}{\mathcal{O}}
\newcommand{\dime}{d}
\newcommand{\domain}{\R^\dime}
\newcommand{\R}{\mathbb{R}}
\newcommand{\N}{\mathbb{N}}
\newcommand{\Pub}{\Phi}
\newcommand{\pub}{\phi}
\newcommand{\D}{\mathcal{D}}
\newcommand{\n}{n}
\newcommand{\Lrn}{\mathrm{A}}
\newcommand{\Unlrn}{\bar{\Lrn}}
\newcommand{\Nsgd}{\text{Noisy-GD}}
\newcommand{\Pgd}{\text{PGD}}
\newcommand{\x}{\mathbf{x}}
\newcommand{\thet}{\theta}
\newcommand{\Thet}{\Theta}
\newcommand{\loss}{\boldsymbol{\ell}}
\newcommand{\rloss}{\bar\loss}
\newcommand{\reg}{\mathbf{r}}
\newcommand{\ltwo}{L2}
\newcommand{\Loss}{\mathcal{L}}
\newcommand{\rLoss}{\Loss}
\newcommand{\err}{\mathrm{err}}
\newcommand{\risk}{\alpha}
\newcommand{\q}{q}
\newcommand{\eps}{\varepsilon}
\newcommand{\epsdp}{\eps*{\mathrm{dp}}}
\newcommand{\epsdd}{\eps*{\mathrm{dd}}}
\newcommand{\del}{\delta}
\newcommand{\cost}{C}
\newcommand{\step}{\eta}
\newcommand{\K}{K}
\newcommand{\ze}{\zeta}
\newcommand{\batch}{\mathcal{B}}
\newcommand{\bsize}{b}
\newcommand{\h}{h}
\newcommand{\Sam}{\mathcal{S}}
\newcommand{\cvx}{\lambda}
\newcommand{\lip}{L}
\newcommand{\Clip}{\mathrm{Clip}}
\newcommand{\Proj}{\mathrm{Proj}}
\newcommand{\smh}{\beta}
\newcommand{\G}{G}
\newcommand{\B}{B}
\newcommand{\conv}{\circledast}
\newcommand{\grad}{\mathbf{g}}

\newcommand{\y}{\mathbf{y}}
\newcommand{\up}{u}
\newcommand{\Up}{U}
\newcommand{\ind}{\mathrm{ind}}
\newcommand{\U}{\mathcal{U}}
\newcommand{\reps}{r}
\newcommand{\delete}{\mathbf{I}_d}
\newcommand{\append}{\D_a}
\newcommand{\sen}{S}
\newcommand{\prox}{\mathrm{prox}}
\newcommand{\publish}{f_{\mathrm{pub}}}
\newcommand{\updreq}{\mathcal{Q}}

\newcommand{\pubs}{p}

\newcommand{\query}{h}
\newcommand{\median}{\mathrm{med}}

\newcommand{\LS}{\mathrm{LS}}
\newcommand{\lsi}{c}
\newcommand{\T}{T}

\newcommand{\eqdef}{\stackrel{\text{def}}{=}}

\newcommand{\approxDP}[1]{\overset{#1}{\approx}}
\newcommand{\KL}[2]{\mathrm{KL}\left(#1\middle\Vert#2\right)}
\newcommand{\Fis}[2]{\mathrm{I}\left(#1\middle\Vert#2\right)}
\newcommand{\Was}[2]{\mathrm{W}_2\left(#1,#2\right)}
\newcommand{\Ren}[3]{\mathrm{R}_{#1}\left(#2\middle\Vert#3\right)}
\newcommand{\Eren}[3]{\mathrm{E}_{#1}\left(#2\middle\Vert#3\right)}
\newcommand{\Gren}[3]{\mathrm{I}_{#1}\left(#2\middle\Vert#3\right)}
\newcommand{\TV}[1]{\mathbf{TV}\left(#1\right)}
\newcommand{\Test}{\text{Test}}
\newcommand{\Testb}{\overline{\text{Test}}}

$$
</p>

Unlearning algorithms aim to remove deleted data's influence from trained models at a cost lower than full retraining. However, we show that **existing Machine Unlearning definitions don't protect the privacy of deleted records in the real world**. The primary reason is that when users request deletion as a function of models trained on their data, records in a database become interdependent---past records influence what future records might be present. So, even retraining a fresh model after deletion of a record doesn't ensure its privacy because by looking at the current behavior due to present records, an attacker can infer the kinds of records that existed in the past, even if they were since deleted. Secondly, we show that unlearning algorithms that cache partial computations to speed up the processing can leak deleted information over a series of releases, violating the privacy of deleted records in the long run. To address these, we propose a sound deletion guarantee and show that the **privacy of existing records is necessary for the privacy of deleted records in the future**. Under this notion, we propose an accurate, computationally efficient, and secure machine unlearning algorithm based on Noisy Gradient Descent. Check out our work's [full version >](https://arxiv.org/abs/2210.08911) (edit: or [my PhD thesis >]({{ site.baseurl }}/assets/pdf/Phd-thesis.pdf)).

## Machine Unlearning

Machine Unlearning is a relatively new notion of deletion privacy that is becoming increasingly important. The need for deletion arises from an evolved understanding that data privacy requires a strong commitment to **self-determinism**, which is _the power of individuals to decide if, to what extent, and how their personal information is collected, used, and shared_<d-cite key="berman2001privacy,rouvroy2009right,debatin2011ethics"></d-cite>. Modern day privacy policies, such as General Data Protection Regulation (GDPR) in the European Union and California Consumer Protection Act (CCPA) in the United States, enforces the principle of self-determination by granting individuals the right to _rectify or erase their data_.

In the era of Deep Learning, information in modern ML models, such as neural networks, is no longer encoded in human interpretable way, making it extremely challenging to selectively edit. The conventional (and fool-proof) way is to modify the underlying dataset and retrain the models from scratch to reflect the changes. However, retraining fresh ML models from scratch every time the dataset changes is computationally expensive. Therefore there is a growing interest in designing cheap **machine unlearning algorithms** for erasing the influence of deleted data from already trained models. To quantify how well an unlearning algorithm erases the requested information in the worst-case,<d-cite key="ginart2019making"></d-cite> propose a DP like $(\eps, \del)$-indistinguishability between the unlearning algorithm's output and that of fresh retraining.


<div id="goal-unlearning" class="definition"><b>($(\eps, \del)$-unlearning).</b>
For a fixed dataset $\D \in \X^*$, remove set $\forget \subset \D$, a randomized learning algorithm $\Lrn$ and a publish function $\publish$, an unlearning algorithm $\Unlrn$ is $(\eps, \del)$-unlearning with respect to $(\D, \forget, \Lrn)$ if for all $S \subset \Y$ where $\Y$ denotes the output space, we have:


$$

\begin{align*}
\prob{}{\publish(\Unlrn(\D, \forget, \Lrn(\D))) \in S}&\leq e^\eps \cdot \prob{}{\publish(\Lrn(\D \setminus \forget)) \in S} + \del, \quad \text{and} \\\\
\prob{}{\publish(\Lrn(\D \setminus \forget)) \in S} &\leq e^\eps \cdot \prob{}{\publish(\Unlrn(\D, \forget, \Lrn(\D))) \in S} + \del.
\end{align*}

$$
</div>

The idea behind this definition is very simple. If there is no way to distinguish the published observable $\publish(\cdot)$ of an unlearned model $\Unlrn(\D, \forget, \Lrn(\D))$ from that of a fresh model $\Lrn(\D \setminus \forget)$ trained without the deleted data-points, then the unlearned model must not contain any information about the deleted records anymore. Several unlearning certifications, that follow the same intuition, have since been proposed and used to certify numerous unlearning algorithms that belong to the family of iterative first-order gradient methods<d-cite key="guo2019certified,izzo2021approximate,neel2021descent,ullah2021machine"></d-cite>.



Almost all real-world applications of machine unlearning need to handle **streaming addition/deletion requests**, first studied by Neel et al., 2021<d-cite key="neel2021descent"></d-cite>. In this setting, a stream of *edit requests*, comprised of a batch of addition, deletion or replacement operations, arrives sequentially.


<div class="definition"><b>(Edit request).</b>
A replacement operation $\langle \ind, \y \rangle \in [\n] \times \X$ on a database ${\D = (\x_1, \cdots, \x_n) \in \X^\n}$ performs the following modification:


$$

\begin{equation}
\D \circ \langle \ind, \y \rangle = (\x*1, \cdots, \x*{\ind-1}, \y, \x*{\ind+1}, \cdots, \x*\n).
\end{equation}

$$

An edit request $\up = \{\langle \ind_1, \y_1\rangle, \cdots, \langle \ind_\reps, \y_\reps\rangle \}$ on $\D$ is defined as batch of $\reps \leq \n$ replacement operations modifying distinct indices atomically, i.e.


$$

\begin{equation}
\D \circ \up = \D \circ \langle \ind*1, \y_1\rangle \circ \cdots \circ \langle \ind*\reps, \y\_\reps\rangle,
\end{equation}

$$
where $\ind_i \neq \ind_j$ for all $i \neq j$.
</div>

<figure id="edit_request">
<picture>
    <source srcset="{{ site.baseurl }}/assets/img/blog/ForgetUnlearning/edit_request.png" media="(min-width: 768px)" />
    <img srcset="edit_request.png" alt="Example of an edit request" />
    <figcaption>Depiction of an edit request modifying a dataset.</figcaption>
</picture>
</figure>

<aside markdown="1">
Over a period of time, some individuals may request additions, while others may request deletions. We model this using batched replacement edits and handle any disparity in the number of additions and deletions by inserting or replacing dummy records $\bot$.
</aside>




The *data curator*, characterized by a pair of learning and unlearning algorithms $(\Lrn, \Unlrn)$, executes the learning algorithm $\Lrn$ on the initial dataset $\D_0$ during the setup stage before the arrival of the first edit request to generate the initial model $\Theta_0 = \Lrn(\D_0)$. Thereafter, at any edit step $i \geq 1$, to reflect an edit request $\forget_i$ that transforms the dataset $\D_{i-1} \circ \forget_i \rightarrow \D_i$, the curator runs the unlearning algorithm $\Unlrn$ to get the next model $\Theta_i = \Unlrn(\Theta_{i-1}, \forget_i, \D_{i-1})$. Furthermore, the curator keeps the (un)learned models $(\Thet_i)_{i\geq0}$ secret, only releasing corresponding *publishable objects* $\pub_i = \publish(\Thet_i)$, generated using some publish function $\publish$. Such publishable objects could be model predictions on an external dataset, noisy model releases, or any downstream model use.


<figure id="unlearning_setup_advanced">
<picture>
    <source srcset="{{ site.baseurl }}/assets/img/blog/ForgetUnlearning/unlearning_setup_advanced.png" media="(min-width: 768px)" />
    <img srcset="unlearning_setup_advanced.png" alt="Adaptive Unlearning Setup" />
    <figcaption>Detailed visualization of the streaming unlearning setup.</figcaption>
</picture>
</figure>

<aside markdown="2">
    Edit requests $\up_i$ depend on (a subset of) the history of the observable objects $(\phi_0, \cdots, \phi_{i-1})$ published in the past. The update requester $\updreq$ (introduced a bit later) models such arbitary dependences among requests that may arise in a real world setting.
</aside>

### Adaptive Unlearning

In real world, **deletion requests are often adaptive in nature**, influenced by the behavior of prior published outcomes. For instance, security researchers may demonstrate privacy attacks targeting a minority subpopulation on publicly available models, causing people in that subpopulation to request deletion of their information from training data. Gupta et al., 2021<d-cite key="gupta2021adaptive"></d-cite> model such an interactive environment through an *adaptive update requester*. We provide the following generalized definition of Gupta et al., 2021<d-cite key="gupta2021adaptive"></d-cite>'s update requester and describe its interaction with a data curator in <a href="#goal-interact">Algorithm 1</a>.


<div id="dfn:updreq" class="definition"><b>(Update requester<d-cite key="gupta2021adaptive"></d-cite>).</b>
At any step $i \in \N$, an update requester $\updreq$ inputs a subset of interaction history $(\phi_0, \up_1, \phi_1, \cdots, \up_{i-1}, \phi_{i-1})$ between themself and the curator $(\Lrn, \Unlrn)$ to generate the subsequent edit request $\up_i$. That is, for an ordered set of integers $(s_1, \cdots, s_j) \subset [i]$ denoting all the steps before $i$ that requester $\updreq$ can observe, the edit request $\up_i$ generated by $\updreq$ on interaction with the curator is given by


$$

\begin{equation}
\up*i = \updreq(\underbrace{\pub*{s*1}, \pub*{s*2}, \cdots, \pub*{s*j}}*{\eqdef \pub*{\vec{s} < i}}; \underbrace{\up_1, \up_2, \cdots, \up*{i-1}}_{\eqdef \up_{< i}}).
\end{equation}

$$

</div>

<aside markdown="3">
    The update requester $\updreq$ encapsulates the curator's interactions with the entire collection of users---with all their unregulatable communications and complex interdependences arising in the real-world outside of view.
</aside>

It is useful to consider the sequence of indices $\vec{s}$ as the <b>planned versions of publishable outcomes to be released</b> as and when generated by the curator. Outcomes at these time steps will have an influence on the world, which in-turn will affect the subsequent requests and corresponding releases.


<figure id="adaptive_requester">
<picture>
    <source srcset="{{ site.baseurl }}/assets/img/blog/ForgetUnlearning/adaptive_requester.png" media="(min-width: 768px)" />
    <img srcset="adaptive_requester.png" alt="Example of an edit request" />
    <figcaption>Visualization of an adaptive edit requester $\updreq$ that looks at a (subset of) past outcome releases to (stochastically) determine the next edit request to issue.</figcaption>
</picture>
</figure>

The above figure depicts an example of an adaptive requester $\updreq$ modelling the interactive nature of the real-world edit requests inserting, deleting, or replacing entries in response to past outcomes released by the curator $(\Lrn, \Unlrn, \publish)$.



<pre id="read-interact" style="display:none;">
\begin{algorithm}
\caption{Interacting curator $(\mathrm{A}, {\mathrm{\bar A}}, f_{\mathrm{pub}})$ \& requester $\mathbb{Q}$}
\begin{algorithmic}
\REQUIRE dataset $D_0 \in \mathcal{X}^n$, observable indices $\vec{s} \in \mathbb{N}^{p}$.
\STATE {Initialize $\Theta_0 \leftarrow \mathrm{A}(D_0)$}
\STATE {Publish $\phi_0 \leftarrow f_{\mathrm{pub}}(\Theta_0)$}
\FOR{$i = 1, 2, \cdots$}
\STATE {Get next request $u_i \leftarrow \mathbb{Q}(\phi_{\vec{s}<i}; u_{<i})$}
\STATE {Update model $\Theta_i \leftarrow {\mathrm{\bar A}}(D_{i-1}, u_i, \Theta_{i-1})$}
\STATE {Publish $\phi_i \leftarrow f_{\mathrm{pub}}(\Theta_i)$}
\STATE {Update dataset $D_i \leftarrow D_{i-1} \circ u_i$}
\ENDFOR
\end{algorithmic}
\end{algorithm}
</pre>
<div id="goal-interact"></div>
<script type="text/javascript">
    var code = document.getElementById("read-interact").textContent;
    var parentEl = document.getElementById("goal-interact");
    var options = {
        lineNumber: true
    };
    pseudocode.render(code, parentEl, options);
</script>

We refer to a requester $\updreq$ that never gets to see any of the past releases (i.e., $\pub_{\vec{s} <i} = \emptyset$ for all $i \in \N$) as being **non-adaptive**. Similarly, we consider a requester that gets to see all the past releases (i.e., $\pub_{\vec{s} < i} = (\pub_0, \cdots, \pub_{i-1})$) as being **fully-adaptive**. A middle-ground is an update requester that gets to see at most $\pubs$ releases ever (i.e., $\vert\vec{s}\vert \leq \pubs$), which we say is **$\pubs$-adaptive**.



<div id="dfn:adaptive_unlearning" class="definition"><b>(Adaptive Machine unlearning<d-cite key="neel2021descent,gupta2021adaptive"></d-cite>).</b>
We say that an algorithm $\Unlrn$ is a <b>non-adaptive $(\eps, \del)$-unlearning</b> algorithm for $\Lrn$ under a publish function $\publish$, if for all initial datasets $\D_0 \in \X^\n$ and all non-adaptive requesters $\updreq$, the following condition holds. For every edit step $i \geq 1$, and for all generated edit sequences ${\up_{\leq i} \eqdef (\up_1, \cdots, \up_i)}$,


$$

\begin{equation}
\label{eqn:unlearn*neel}
\publish(\Unlrn(\D*{i-1}, \up*i, \Theta*{i-1})) \big|_{\up_{\leq i}} \approxDP{\eps,\del} \publish(\Lrn(\D_i)),
\end{equation}

$$

where $X\approxDP{\eps,\del}Y$, for any two random variables $X, Y$, denotes $(\eps, \del)$-DP like indistinguishability between them. If the above equations holds for all fully-adaptive requesters $\updreq$, we say that $\Unlrn$ is an <b>$(\eps, \del)$-adaptive-unlearning</b> algorithm for $\Lrn$.
</div>



## Flaws in Unlearning

<p class="epigraph">
  "The controller shall have the obligation to erase personal data without undue delay where ... the data subject withdraws consent ..."
  <span class="epigraph-source">— Right to be Forgotten, Article 17(1)(b), GDPR.</span>
</p>
<style>
.epigraph {
  text-align: center;
  font-style: italic;
  margin-bottom: 20px;
  text-indent: 2em; /* Indent the epigraph text */
}

.epigraph-source {
  font-size: smaller;
  text-align: right;
}
</style>

Now we expose some critical limitations of prior unlearning certifications when it comes to upholding the ``Right to be Forgotten''. These limitations manifest under a threat model that is actualized in reality. We highlight multiple reasons why both adaptive and non-adaptive machine unlearning, as described in its <a href="#goal-unlearning">definition</a>, are inadequate in addressing the following threat model.


### Threat Model

The RTBF guidelines in GDPR and CCPA require *permanent deletion* of personal information, *regardless of its form*, and without *undue delay after receipt of a legitimate deletion request from the user*. Considering that data curators are given a grace period to process deletion requests, we must assume in our threat model that an attacker targeting a record deleted at the $i^\mathrm{th}$ step can *only observe releases by the curator after deletion*. In other words, the **attacker sees all post-deletion releases ${\pub_{\geq i} \eqdef (\pub_i, \pub_{i+1}, \cdots)}$.**


<aside id="4">
    Our threat model doesn't include attacks which involve comparing releases before and after deletion (such as Chen et al.'s<d-cite key="chen2021machine"></d-cite>). Succeeding in such attacks technically do not violate RTBF as the model that was released before the issue of the deletion request was the true source of information.
</aside>

Furthermore, we make another assumption about the attacker that makes it considerably stronger---**the attacker has arbitrary knowledge about how some users may react to certain published outcomes**. This assumption is central to the attack vector arising due to the adaptive nature of the real-world. Even though our attacker can only see published objects *after* the deletion request has been processed, she can try and deduce what was deleted by looking for patterns among other data records arising due to the **original presence** of the deleted data based on her understanding of the world. That is to say, our attacker knows a little about how people react to things in general, without actually knowing what they reacted to and how in this specific interactions with the curator. Since $\updreq$ encapsulates all the interactions the entire collection of users can have with the curator, our assumption is simple---**attacker knows $\updreq$**.


<div class="goal">
    The goal of the attacker is to figure out what was deleted at step $i$ by the request $\up_i$.
</div>
<style>
.goal {
  display: block;
  margin: 16px 0;
  padding: 12px;
  background-color: lightblue;
  border-left: 4px solid darken(lightblue, 10%);
  border-radius: 4px;
  transition: transform 0.2s ease, box-shadow 0.2s ease;
}
.goal:hover {
  transform: scale(1.02);
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.15);
}
.goal::before {
  content: "Goal: ";
  font-weight: bold;
  margin-right: 6px;
}
</style>

### Failure under Adaptivity

Machine Unlearning guarantees are **not trustworthy** when the addition/deletion requests are *adaptive*. Before we dive into the technical reasons, let's look at a couple of illustrative examples to identify the root cause of the failure.



<div class="example">
    Consider the case of <a href="https://www.instagram.com/samdoesarts">Sam Yang</a>, a renowned Canadian digital illustrator with hundreds of artworks posted on Instagram and a follower count exceeding 2.1 million. <a id="ref-a" href="#fig-a">Figure (a)</a> below shows an original artwork by Sam titled "Night scene with Kara."

    <figure>
        <div style="display: flex; justify-content: space-between; gap: 10px;">
            <div id="fig-a" style="text-align: center; width: 32%;">
                <picture>
                    <img src="{{ site.baseurl }}/assets/img/blog/ForgetUnlearning/night_scene_with_kara(resized).jpeg" alt="Original 'Night scene with Kara'" style="width: 100%;">
                </picture>
                <figcaption>(<a href="#ref-a">a</a>)) Original "Night scene with Kara".</figcaption>
            </div>
            <div id="fig-b" style="text-align: center; width: 32%;">
                <picture>
                    <img src="{{ site.baseurl }}/assets/img/blog/ForgetUnlearning/samdoesarts_model_v2.png" alt="A stylistic copy via the SamDoesArt-V3 model" style="width: 100%;">
                </picture>
                <figcaption>(<a href="#ref-b">b</a>) 1st gen copy <a href="https://huggingface.co/Sandro-Halpo/SamDoesArt-V3">SamDoesArt-V3</a>.</figcaption>
            </div>
            <div id="fig-c" style="text-align: center; width: 32%;">
                <picture>
                    <img src="{{ site.baseurl }}/assets/img/blog/ForgetUnlearning/a_free_samdoesart_model_v0.png" alt="A second-generation stylistic copy" style="width: 100%;">
                </picture>
                <figcaption>(<a href="#ref-c">c</a>) 2nd gen copy.</figcaption>
            </div>
        </div>
        <figcaption>A real-life example where even retraining does not ensure total deletion.</figcaption>
    </figure>

    <p>Recently, a Redditor utilized Stability AI, a deep-learning platform for creating custom generative models, to train a stable-diffusion model<d-cite key="zhang2023adding"></d-cite>—denoted as $\Theta_0$—using a dataset $\D_0$ exclusively comprised of Sam's artworks. The Redditor then posted examples of generated images online (denoted as $\pub_0$), such as the one shown in <a id="ref-a" href="#fig-b">Figure (b)</a> above. Soon after, internet users began generating and <a href="https://www.reddit.com/r/StableDiffusion/search/?q=samdoesarts">posting stylistic copies</a> on Reddit.</p>

    <p>Infuriated by this, Sam Yang, along with several other illustrators, filed a class-action lawsuit against Stability AI and Midjourney for copyright infringement<d-cite key="shanti2024midjourney"></d-cite>. Since the uploads of stylistic copies and the deletion of original artworks occurred almost simultaneously, let's refer to these actions collectively as a <it>single batched edit request</it> $\up_1$. In response to the lawsuit, an anonymous Redditor curated a dataset $\D_1 = \D_0 \circ \up_1$ that contained only the stylistic copies, excluding Sam's originals, to circumvent copyright infringement<d-cite key="reddit2022post"></d-cite>. <a id="ref-a" href="#fig-c">Figure (c)</a> above depicts a second-generation stylistic copy $\pub_1$ produced by a model $\Theta_1 = \Lrn(\D_1)$ trained solely on synthetic images. It is evident that Sam's distinctive art style persists, even after retraining from scratch without his original works!</p>
</div>

To understand the cause of this problem under adaptivity more analytically, let's consider a simplified variant of this problem.

<div class="example">
For a data domain $\X = \{-2, -1, 1, 2\}$, consider the following learning and unlearning algorithms $(\Lrn, \Unlrn)$. For any dataset $\D \subset \X$ and any subset $S \subset \D$ of records to be deleted,


$$

\begin{equation}
\Lrn(\D) = \sum*{\x \in \D} \x, \quad \text{and} \quad \Unlrn(\D, S, \Lrn(\D)) = \sum*{\x \in \D \setminus S} \x,
\end{equation}

$$

Note that the unlearning algorithm $\Unlrn$ perfectly imitates the learning algorithm $\Lrn$ as for any deletion request $\up \subset \D$, we have $\Unlrn(\D, \up, \Lrn(\D)) = \Lrn(\D \setminus \up)$. Now consider two neighboring datasets $\D_{-1} = \{-2, -1, 2\}$, $\D_{1} = \{-2, 1, 2\}$ and the following dependence between the learned model $\Lrn(\D)$ and deletion request $\up$:


$$

\begin{equation}
\up = \begin{cases} \{\x \in \X | \x < 0\} &\text{if}\ \Lrn(\D) < 0, \\ \{\x \in \X | \x \geq 0\} &\text{otherwise.}\end{cases}
\end{equation}

$$

<p>Knowing this dependence, an attacker can distinguish whether $\D$ is $\D_{-1}$ or $\D_{1}$ by looking solely at $\Unlrn(\D, \x, \Lrn(\D))$. This is because if $\D = \D_{-1}$, then the output after deletion is positive, and if $\D = \D_{1}$ the output is negative. Note that even though $\Unlrn$ perfectly imitates retraining via $\Lrn$ and the attacker does not observe either the model $\Lrn(\D)$ or the request $\up$, she can still ascertain the identity ($-1$ or $1$) of a deleted record. This example demonstrates two things:</p>

    <p><b>A)</b> Adaptive requests can cause the curator's dataset to have patterns specific to the identity of a target record being deleted.</p>

    <p><b>B)</b> An attacker knowing the relationship between unobserved releases and deletion requests can infer the identity of the target record by observing only the unlearned model, even if the curator did full retraining.</p>
</div>

<aside id="5">
    Note: An edit request containing only deletion operations, i.e., replacement with $\bot$, can be equivalently denoted as a set $\up \subset \D$ consisting of the records at indices stated in the edit request.
</aside>


Given the adaptive nature of the real-world interactions, several data deletion definitions in the literature, such as those proposed by <d-cite key="ginart2019making"></d-cite>, <d-cite key="guo2019certified"></d-cite>, and <d-cite key="sekhari2021remember"></d-cite>, that quantify data-deletion based on indistinguishability from retraining do not provide reliable certifications for the *"Right to be Forgotten"*. Even the <a href="#dfn:adaptive_unlearning">Adaptive unlearning definition</a>, which was specifically designed to ensure RTBF under adaptive deletion requests, fails catastrophically under adaptivity!


### Failure with Hidden-States

Both adaptive and non-adaptive unlearning guarantees are bounds on information leakage about a deleted record through a <i>single released output</i>. However, a <a href="#threat-model">real-world adversary</a> would most likely observe multiple (potentially all) releases $\phi_{>i}$ after deletion. This can lead to another violation of RTBF, even when edit <i>requests are non-adaptive</i>.

<figure id="secret_states">
<picture>
    <img src="{{ site.baseurl }}/assets/img/blog/ForgetUnlearning/secret_states.png" alt="The problem with deletion when using secret states." style="width: 100%;">
    <figcaption>Demonstrating the problem with secret states in $(\eps,\del)$-unlearning as <a href="#dfn:adaptive_unlearning">defined here</a>. Consider a data-point deleted by first edit request $\up_1$. Secret states $\Theta_1, \Theta_2, \cdots$ can carry unbounded information about this deleted record while $\publish$ ensures each individual release carries no more than $(\eps, \del)$ amount of information about deleted records. Since each future release can reveal this much <i>new</i> information, the total leakage composes without any limits.</figcaption>
</picture>
</figure>

This vulnerability arises because many unlearning definitions permit the curator to store secret models<d-cite key="ginart2019making,gupta2021adaptive"></d-cite> while requiring indistinguishability only over the output of a publishing function $\publish$. These secret models may propagate encoded information about records even after their deletion from the dataset. So, every subsequent release by an unlearning algorithm can reveal <i>new information</i> about a record that was purportedly erased multiple edits earlier. That is, an adversary that <b>observes enough future releases</b> may learn <i>everything</i> about the deleted record through the <i>post-deletion</i> releases!


### Incompleteness

Finally, the <a href="#goal-unlearning">$(\eps, \del)$-unlearning guarantee</a> disregards perfectly valid deletion algorithms. For instance, an algorithm $\Unlrn$ that outputs a <i>fixed untrained model</i> $\theta \in \Y$ in response to any deletion request $\forget$ is a valid deletion algorithm because its output contains no information about the deleted records in $\forget$. However, since its fixed output $\theta$ is easy to tell apart from that of retraining done by any sensible learning algorithm $\Lrn$, the algorithm $\Unlrn$ cannot satisfy <a href="#goal-unlearning">this definition</a>, or <a href="#dfn:adaptive_unlearning">this definition</a>, or any other unlearning variants in literature<d-cite key="sekhari2021remember,ginart2019making,guo2019certified"></d-cite>. In other words, <b>machine unlearning guarantees are incomplete quantifiers of data deletion</b>.


## Trustworthy Data-Deletion


To address these issues with existing notions of unlearning, let's introduce a new definition of deletion privacy.

<div id="def:deletion" class="definition"><b>($(\eps, \del)$-deletion privacy).</b>
An algorithm pair $(\Lrn, \Unlrn)$ satisfies $(\eps, \del)$-deletion privacy if for all datasets $\D$ and all edit requests $\forget$ (potentially chosen reactively after seeing $\Lrn(\D)$), we have that for all records $\x \in \D$ that gets deleted by $\forget$, there exists a random variable $\Theta_\x$ that is independent of the data-point $\x$ such that


$$

\begin{equation}
\forall S \in \Y, \quad \prob{}{\Unlrn(\D, \forget, \Lrn(\D)) \in S} \leq e^\eps \cdot \prob{}{ \Theta\_\x \in S} + \del.
\end{equation}

$$
</div>

<aside id="6">
    Think of $\Theta_\x$ as a model that is free of any influence due to the <b>original presence</b> of $\x$ in $\D$. A satisfying construction of $\Theta_\x$ could be the output from an alternate run of the interatction between curator $(\Lrn, \Unlrn)$ and the requester $\updreq$ on a neighbouring dataset $\D' = \D \setminus \{\x\}$ that never contained $\x$.
</aside>


Firstly, using secret states that depend on deleted records to speed up unlearning can violate RTBF. To prevent such violations, our definition directly quantify deletion for an unlearned model rather than after applying any publish function, i.e., setting $\publish(\theta) = \theta$


Secondly, as demonstrated previously, adaptive requests can encode patterns specific to a target record which persists in the dataset even after deletion of the target record, making indistinguishable-from-retraining based deletion certifications unreliable. Our definition accounts for the worst-case influence adaptive requests by measuring the indistinguishability of an unlearning mechanism's output from that of some constructed random variable $\Theta_\x$ that must be independent of the deleted record <i>by design</i>.

<b>Soundness of $(\eps, \del)$-deletion privacy.</b>  The above <a href="#def:deletion">definition</a> reliably safeguards the <i>"Right to be Forgotten"</i> as the random variable $\Theta_\x$ stays independent of the deleted record $\x$ by design, even when the update requester $\updreq$ is fully-adaptive. When an attacker aims to identify a record in $\D$ that is being deleted in edit request $\up$, the inequality in the definition ensures that any observer of the unlearned model $\Unlrn(\D, \up, \Lrn(\D))$ cannot be overly certain that the observation was <i>not</i> $\Theta_\x$ instead. Hence, the unlearned model itself must possess minimal information regarding the deleted record $\x$. This argument allows us to establish the following guarantee of soundness.

<div id="thm:soundness" class="theorem">
    If the algorithm pair $(\Lrn, \Unlrn)$ satisfies $(\eps, \del)$-deletion privacy guarantee under all $\pubs$-adaptive requesters, then any attacker ${\text{MI}: \Y^* \rightarrow \{0,1\}}$ observing only the post-deletion models $\Theta_{\geq i} = (\Theta_i, \Theta_{i+1}, \cdots) \in \Y^*$ after processing the request $\up_i$ has an advantage


$$

    \begin{equation}
    	\text{Adv}(\mathrm{MI})  \eqdef  \prob{}{\mathrm{MI}(\Theta_{\geq i})  =  1 \big| \x}  - \prob{}{\mathrm{MI}(\Theta_{\geq i})  =  1 \big| \x'}
    \end{equation}
    $$

for disambiguating between two possible values $\x, \x' \in \X$ of a record in $\D_{i-1}$ deleted by request $\up_i$ bounded as follows.

    $$
    \begin{equation}
    	\label{eqn:adv_sound}
    	\text{Adv}(\mathrm{MI}) \leq e^\eps - 1 + 2\del.
    \end{equation}
    $$

</div>

### Link to Differential Privacy

A differential privacy guarantee on $\Lrn$ and $\Unlrn$ sets a limit on the information present in an unlearned model regarding individual records that <i>remain in the dataset</i>. However, our concept of deletion privacy specifically restricts the information concerning <i>only the deleted records</i>. These two notions are <b>orthogonal</b>---an algorithm pair $(\Lrn, \Unlrn)$ can satisfy $(\epsdd,\del)$-deletion privacy without providing $(\epsdp,\del)$-differential privacy for any $\infty > \epsdp > 0$. These two notions are also <b>compatible</b>---an algorithm pair $(\Lrn, \Unlrn)$ can simultaneously satisfy $(\epsdp,\del)$-differential privacy and $(\epsdd, \del)$-deletion privacy with $\epsdd \ll \epsdp$ for non-adaptive update requesters $\updreq$.

However, the two types of privacy certifications are connected---an $(\epsdp,\del)$-differential privacy is both a <b>necessary</b> and a <b>sufficient</b> condition to ensure that a non-adaptive $(\epsdd, \del)$-deletion privacy guarantee for $(\Lrn, \Unlrn)$ extends to adaptive settings (which is significantly more challenging) with a <i>graceful degradation</i> in deletion certification.

#### Sufficiency

For the <b>sufficient</b> part, note that when $\Lrn$ and $\Unlrn$ are both differentially private, they prevent an adaptive requester from establishing dependencies between records in the curator's dataset. By leveraging this property, we establish a reduction from adaptive to non-adaptive deletion privacy only assuming that $\Lrn$ and $\Unlrn$ also satisfy $(\epsdp,\del)$-differential privacy.

<figure id="dp_dependence">
<picture>
    <img src="{{ site.baseurl }}/assets/img/blog/ForgetUnlearning/dp_deletion.png" alt="Relationship between diffential and deletion privacy." style="width: 100%;">
    <figcaption>Differential privacy guarantee controls how much inter-dependence an adaptive userbase can introduce between the entries in the dataset over a period of time (or, in our case, number of planned releases $\pubs$).</figcaption>
</picture>
</figure>

<div id="thm:reduction" class="theorem">
    If an algorithm pair $(\Lrn, \Unlrn)$ satisfies $(\epsdd, \del)$-deletion privacy under all non-adaptive requesters and is also $(\epsdp, \del)$-differentially private, then pair $(\Lrn, \Unlrn)$ also satisfies $(\epsdd', (\pubs + 2)\del)$-deletion privacy under all $\pubs$-adaptive requesters, for

    $$
    \begin{equation}
        \epsdd' = \epsdd + \pubs \epsdp(e^\epsdp - 1) + \epsdp\sqrt{2\pubs \log(1/\del)}.
    \end{equation}
    $$

</div>

<aside id="7">
This theorem is essentially the same as composition of the $\pubs$-fold $(\epsdp, \del)$-differential privacy guarantees with a singular $(\epsdd,\del)$-deletion privacy guarantee. 
</aside>

When $\epsdd \ll \epsdp$, note that the adaptive deletion privacy guarantee is barely larger than just the DP bound on the total information revealed about the deleted record through the past releases before the deletion request was issued. That is to say, when $\Lrn, \Unlrn$ are differentially private, any deletion privacy guarantee under non-adaptivity <i>gracefully reduces to deletion-privacy under adaptivity</i> with the worst-case degradation being equivalent to the composition of the two bounds.

The great thing about this theorem is that it simplifies the certification process for unlearning algorithms to ensure RTBF compliance. Under the assumption that deletion requests are independent of previous releases, i.e., non-adaptive, showing that the algorithm pair $(\Lrn, \Unlrn)$ is both differentially private and provides deletion privacy is sufficient!

#### Necessity

In order to guarantee deletion privacy for erased records in the real-world setting, it is <b>necessary</b> for the unlearning algorithm to uphold the privacy of records that remain undeleted. This is because the only effective means of preventing the adaptive world from reacting to the presence of a target record before deletion is by ensuring it never becomes aware of its existence.

<div id="thm:necessary" class="theorem">
Let $\Test: \Y \rightarrow \{0, 1\}$ be a membership inference test for $\Lrn$ to distinguish between neighboring datasets $\D, \D' \in \X^\n$. Similarly, let $\Testb: \Y \rightarrow \{0, 1\}$ be a membership inference test for $\Unlrn$ to distinguish between $\bar\D, \bar\D' \in \X^\n$. If $\text{Adv}(\Test) > \Delta_\Lrn$ and $\text{Adv}(\Testb) > \Delta_\Unlrn$, then the pair $(\Lrn, \Unlrn)$ <i>cannot</i> satisfy $(\eps, \del)$-deletion privacy under all $1$-adaptive requester for any

$$
    \begin{equation}
        \eps < \log(1 + \Delta_\Lrn \cdot \Delta_\Unlrn - 2\del).
    \end{equation}
$$

</div>

<aside id="8">
    In layman terms, this theorem says that if membership inference attacks are possible for the records not being deleted (i.e., no differential privacy), then there exists some adaptive requester for which it is impossible to guarantee $(\epsdd, \del)$-deletion privacy with tiny values of $(\epsdd, \del)$.
</aside>

This theorem shows that without membership privacy, which is ensured by differential privacy, it isn't possible to provide any non-trivial deletion privacy guarantees under adaptivity. Therefore, for building reliable unlearning algorithms, we should look towards differentially-private algorithms.

#### Clarifying the need for Differential Privacy.

It is important to emphasize that we are not advocating for unlearning solely through differentially private mechanisms, as they uniformly limit the information content of all records, whether deleted or not. Instead, an effective unlearning algorithm should offer two distinct information retainment bounds: one for the records currently present in the dataset, provided by a differential privacy guarantee, and a significantly smaller bound for the records previously deleted, ensured through a non-adaptive deletion privacy guarantee. Together, these two bounds ensure privacy of deleted records under adaptivity, thanks to <a href="#thm:reduction">this theorem</a>.

## Conclusion

In this blog post, we highlighted that existing unlearning certifications in literature are unreliable in real-world scenarios, mainly due to their failure in handling adaptive deletion requests and because they permit unchecked propagation of deleted information through internal states as they only measure indistinguishability for outputs. To mitigate these, we proposed a new deletion guarantee that safeguards the ``Right to be Forgotten'' in a provably secure way. We also showed the importance of protecting the privacy of existing records in order to ensure privacy of deleted records under adaptive deletions, and established connections between deletion privacy and differential privacy.
