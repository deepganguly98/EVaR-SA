# EVaR-SA
Code Submission for TMLR paper: "Risk-Seeking Reinforcement Learning via Multi-Timescale EVaR Optimization"

Abstract: Tail-aware objectives shape agents' behavior when navigating uncertainty and can depart from risk-neutral scenarios. Risk measures such as Value at Risk (VaR) and Conditional Value at Risk (CVaR) have shown promising results in reinforcement learning. In this paper, we study the incorporation of a relatively new coherent risk measure, Entropic Value at Risk (EVaR), as a high-return, risk-seeking objective that the agent seeks to maximize. We propose a multi-timescale stochastic approximation algorithm to seek the optimal parameterized EVaR policy. Our algorithm enables effective exploration of high-return tails and robust gradient approximation to optimize the EVaR objective.  We analyze the asymptotic behavior of our proposed algorithm and rigorously evaluate it across various discrete and continuous benchmark environments. The results highlight that the EVaR policy achieves higher cumulative returns and corroborate that EVaR is indeed a competitive risk-seeking objective for RL.


Citation:
@article{
ganguly2025riskseeking,
title={Risk\nobreakdash-Seeking Reinforcement Learning via Multi\nobreakdash-Timescale {EV}aR Optimization},
author={Deep Kumar Ganguly and Ajin George Joseph and Sarthak Girotra and Sirish Sekhar},
journal={Transactions on Machine Learning Research},
issn={2835-8856},
year={2025},
url={https://openreview.net/forum?id=4nbEgNDsii},
note={}
}
