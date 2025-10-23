# EVaR-SA

Official code repository for the TMLR paper:  
**Risk-Seeking Reinforcement Learning via Multi-Timescale EVaR Optimization**  
by **Deep Kumar Ganguly**, **Ajin George Joseph**, **Sarthak Girotra**, and **Sirish Sekhar**  
(*Transactions on Machine Learning Research*, 2025)

---

## 🧠 Abstract

Tail-aware objectives shape an agent’s behavior when navigating uncertainty and can significantly differ from risk-neutral formulations.  
Risk measures such as **Value at Risk (VaR)** and **Conditional Value at Risk (CVaR)** have been widely used in reinforcement learning (RL).

In this work, we study the use of a relatively new **coherent risk measure**, the **Entropic Value at Risk (EVaR)**, as a **high-return, risk-seeking** objective.  
We propose a **multi-timescale stochastic approximation algorithm** (EVaR-SA) to learn the optimal parameterized EVaR policy.  
Our approach enables efficient exploration of high-return tails and provides a robust gradient approximation for optimizing the EVaR objective.

Theoretical analysis establishes asymptotic convergence and finite-time behavior, while experiments on discrete and continuous control benchmarks demonstrate that EVaR policies achieve higher cumulative returns—confirming that EVaR is a competitive risk-seeking objective in RL.

---

## 📘 Citation

If you find this repository useful, please cite:

```bibtex
@article{
  ganguly2025riskseeking,
  title={Risk-Seeking Reinforcement Learning via Multi-Timescale {EV}aR Optimization},
  author={Deep Kumar Ganguly and Ajin George Joseph and Sarthak Girotra and Sirish Sekhar},
  journal={Transactions on Machine Learning Research},
  issn={2835-8856},
  year={2025},
  url={https://openreview.net/forum?id=4nbEgNDsii}
}
