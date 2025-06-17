Achieved SOTA!
- Previous: 2.634
- AlphaEvolve: 2.63586275
- OpenEvolve: 2.634292402141039 (with assistance)
- Timo Berthold: 2.63591551 (FICO Xpress Solver)
- Me: 2.635983066

![sota](docs/sota.png)

TODO
- generic analyses
- circle packing analyses
- wandb logging
- check for unevaluated (low prio)
- have a think about generations and migrations
  - if you don't improve on the parent score, the generation will not advance
  - the island can be stagnant until a neigbouring island that is improving migrates its elite
- config saving
- map elites: time remaining, diversity, simplicity
- generation and/or program count combined migration rules
