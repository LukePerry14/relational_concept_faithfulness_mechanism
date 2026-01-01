# TODO

- make mu parameters learnable
- allow relation sequences to be learned as well as gamma




# Things to consider moving forward
- what if we only need part of a path? i.e. if the path from a source node is A -> B -> C but we only need A -> B, Then sampled paths will be A -> B -> C but might not match the A -> prototype
  - I assume for now that the relational prototype can still capture this relationship while the gamma values can ablate these values

- we can use a regularisation proportional to the inverse of the sum of gamma values to push gamma values to become large (and therefore ignore content)