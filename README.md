# engagement-level-detection
Automated engagement level detection for [EmotiW'20 challenge](https://sites.google.com/view/emotiw2020/challenge-details).
The study result is represented as Flask web-app.

This repository contains the following modules:
* [Flask app main body](https://github.com/Tat-V/engagement-level-detection/tree/main/flask_app)
* [Saved ML and DL pretrained models](https://github.com/Tat-V/engagement-level-detection/tree/main/models)
  - MobileNet and pretrained faces are borrowed from [repo](https://github.com/HSE-asavchenko/face-emotion-recognition)
  - SVC model was saved as the best engagement regression model during research
* [Prediction](https://github.com/Tat-V/engagement-level-detection/tree/main/prediction) contains useful functions for making predictions on prepared videos
* [Resources](https://github.com/Tat-V/engagement-level-detection/tree/main/resources) contains an example of the table with potential churners information
* [Video parsing](https://github.com/Tat-V/engagement-level-detection/tree/main/video_parsing) contains the functions for video preprocessing
