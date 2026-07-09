# Historical Events: 2013 Alberta Floods and 2013 Queensland Floods (text-only, out-of-domain)

In June 2013, heavy rainfall caused severe flooding across southern Alberta, Canada,
including major flooding in Calgary and High River, prompting large-scale evacuations.
Separately, in January 2013, ex-Tropical Cyclone Oswald brought major flooding to
Queensland, Australia, affecting Bundaberg and Brisbane among other areas. Both events
generated substantial crisis-related social media activity and are represented in the
CrisisLexT6 benchmark dataset as informativeness-labeled tweet collections.

These two events are used in this benchmark as an **out-of-domain, text-only**
generalization test: no U.S. structured gauge data can be paired with them (they are
outside USGS NWIS coverage), so they are used only to test whether the agent's text/RAG
branch generalizes to flood reports outside its US-focused training/knowledge-base
distribution, not for the joint structured+text classification task.
