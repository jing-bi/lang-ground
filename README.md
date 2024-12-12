# Language Grounding

Localize and keep tracking things based on natural
language sepecfication is a good idea but remains challenging due to the scarcity of large-scale annotated datasets.

This repository provides a practical solution to identify relevant objects in the view and continuously tracking them, which is particularly beneficial for robotics applications. 

We offer several demos that showcase the following functionalities:

1. **Lang Localize:** Identify and locate objects based on natural language queries.
2. **Lang Ground (Localize + Track):** Not only find but also continuously track objects of interest

We make sure this repo is open box use and pave the way for other projects.


## 🛠️ Install

```bash
git clone https://github.com/jing-bi/lang-ground.git && cd lang-ground

mamba create -n lang-ground python=3.11
mamba activate lang-ground

pip install -e .
```

