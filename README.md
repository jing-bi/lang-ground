# Language Grounding

Localize and keep tracking things based on natural
language sepecfication is a good idea but remains challenging due to the scarcity of large-scale annotated datasets.

This repository provides a practical solution to identify relevant objects in the view and continuously tracking them, which is particularly beneficial for robotics applications.

We offer several demos that showcase the following functionalities:

## Language Localize
Identify and locate objects based on natural language queries.

Try our demo to experience an interesting use case: When analyzing food items, the model demonstrates contextual understanding:
- For food nearing expiration: Suggests storing in the cabinet
- For expired food: Recommends disposal in the trash can

![langloc](/assets/langloc.jpg)




## Language Ground (Localize + Track)
Not only find but also continuously track objects of interest
<video width="100%" controls>
    <source src="/assets/langgnd.mp4" type="video/mp4">
</video>

The key component for this is above localization plus a elegent implemenetation of a stream SAM2 that support to the lastest version.








We prioritize making this project highly accessible and customizable:

- **Open Box Design:** All components are modular and well-documented for easy understanding
- **Customizable Pipeline:** Easily adapt the system for different use cases
- **Extensible Framework:** Simple integration with other vision or language models


## 🛠️ Install

```bash
git clone https://github.com/jing-bi/lang-ground.git && cd lang-ground

mamba create -n lang-ground python=3.11
mamba activate lang-ground

pip install -e .
```
## Acknowledgments

This project is built upon and inspired by the following repositories:

- [Segment-Anything](https://github.com/facebookresearch/segment-anything-2)
- [Supervision](https://github.com/roboflow/supervision)

## License

This project is licensed under the Apache 2.0 License