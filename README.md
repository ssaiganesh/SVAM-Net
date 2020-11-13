### SVAM: Saliency-guided Visual Attention Modeling 
![svam-fig](/data/svam_fig0.jpg)

### Pointers
- Preprint: https://arxiv.org/pdf/2011.06252.pdf
- Video demonstration: https://youtu.be/SxJcsoQw7KI
- Data: http://irvlab.cs.umn.edu/resources/usod-dataset
- Project page: http://irvlab.cs.umn.edu/visual-attention-modeling/svam

### SVAM-Net Model
- Jointly accommodate bottom-up and top-down learning in two branches sharing the same encoding layers
- Incorporates four spatial attention modules (SAMs) along these learning pathways 
- Exploits coarse-level and fine-level semantic features for SOD at four stages of abstractions
- The bottom-up pipeline (SVAM-Net_Light) performs abstract saliency prediction at fast rates
- The top-down pipeline ensures fine-grained saliency estimation by aresidual refinement module (RRM)

### SVAM-Net Features
- Provides SOTA performance for SOD on underwater imagery 
- Exhibits significantly better generalization performance than existing solutions
- Achieves fast end-to-end inference
	- The end-to-end SVAM-Net : 20.07 FPS in GTX-1080, 4.5 FPS on Jetson Xavier
	- Decoupled SVAM-Net_Light: 86.15 FPS in GTX-1080, 21.77 FPS on Jetson Xavier

### USOD Dataset
- A new challenging test set for benchmark evaluation of underwater SOD models
- Contains 300 natural underwater images and ground truth labels
- Can be downloaded from: http://irvlab.cs.umn.edu/resources/usod-dataset

#### Bibliography entry:
	
	@article{islam2020svam,
	    title={{SVAM: Saliency-guided Visual Attention Modeling 
	    	    by Autonomous Underwater Robots}},
	    author={Islam, Md Jahidul and Wang, Ruobing and de Langis, Karin and Sattar, Junaed},
	    journal={arXiv preprint arXiv:2011.06252},
	    year={2020}
	}

