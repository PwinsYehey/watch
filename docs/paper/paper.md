% WATCH: Wakefulness Assessment through Tracking of Cues & Head
% Real-time alertness monitoring from the eyes, mouth, head and posture
%% Prince Herrera, Ormin Cariaso, John Casili, Christenzien Lalimo

> **Abstract.** Drowsiness at the controls of a vehicle or aircraft shows in the eyes, mouth, head and posture well before control is lost, yet camera-based monitors usually assume a visible face and a capable computer. This paper presents WATCH, a real-time alertness monitor that runs on a Raspberry Pi 4. Two small convolutional networks classify eye and mouth crops, and each has a third class, *blocked*, for crops it cannot read. Their outputs are combined with face-landmark geometry, head pose and upper-body posture, calibrated to each user, and escalated over time from warning to critical. When a visor or mask hides the face, the monitor relies on posture and asks the user to check in. Trained on four public datasets and tested on people not seen in training, the eye and yawn models reach 96.5% and 95.0% accuracy. On photos of ten motorcycle riders in helmets, closed-eye readings rose from 3.7% in alert sessions to 38.2% in drowsy ones. The full monitor runs at 4 to 6 frames per second on the Pi.

> **Keywords:** drowsiness detection, driver and pilot monitoring, eye state classification, yawn detection, occlusion, embedded systems, Raspberry Pi.

# 1. Introduction

Drowsiness slows reaction time and narrows attention long before a person falls asleep. Behind the wheel or at the controls of an aircraft, a few seconds of microsleep is enough to cause an accident. The early signs are visible from the outside: the eyelids stay closed for longer, yawns become frequent, the head drops forward or to the side, and the body slumps in the seat. A camera that watches for these signs can warn the operator before their attention fails.

WATCH (Wakefulness Assessment through Tracking of Cues & Head) is a real-time alertness monitor built on this idea. It reads the eyes, mouth, head and upper body from a single camera, combines these cues over time, and raises an alert at two levels, warning and critical. It is built for low-cost hardware. The complete system runs on a Raspberry Pi 4 with a Camera Module 3 NoIR, and its dashboard is viewed in a browser on another device.

Two practical problems shaped the design. First, faces differ, so a fixed threshold that suits one person misfires on another. Each session therefore begins with a short calibration of the user's normal eye opening, mouth and head pose. Second, the face is not always visible. A tinted visor, sunglasses or a mask can hide the eyes and mouth. When that happens, WATCH does not go silent. It relies on body posture instead, and asks the user to check in when the eyes have been unreadable for a minute.

The main contributions of this work are:

- Two lightweight convolutional networks, one for eye state (111,043 parameters) and one for yawning (14,737 parameters), trained on four public datasets with subject-wise splits. Each has a third class, *blocked*, for crops that cannot be read.
- A fusion method that combines the network outputs with face-landmark geometry (eye and mouth aspect ratios, head pitch and roll) and per-user calibration, escalating from warning to critical based on how long and how often signs appear.
- Upper-body posture monitoring (slumping, leaning, nodding off and staying still) and a face-covered mode, so users in helmets and masks are still monitored.
- A working deployment on a Raspberry Pi 4 and in the browser, with session reports that record every alert and its cause for later review.

The rest of the paper is organized as follows. Section 2 reviews related work. Sections 3 and 4 describe the data and how it was prepared. Sections 5 and 6 cover the models and their training. Section 7 explains the real-time alertness logic, and Section 8 the deployment. Section 9 presents the results, Section 10 discusses limitations, and Section 11 concludes.

# 2. Related Work

## 2.1 Approaches to drowsiness detection

A recent review groups drowsiness detection systems into four types: image-based, biological, vehicle-based and hybrid [1]. Biological methods measure brain, heart or muscle activity and are accurate, but they need sensors worn on the body. Vehicle-based methods watch steering and lane position. They need no contact, but they depend on the vehicle and react only once driving has already degraded. Image-based methods use a camera to watch the face and body. They need no contact, work in any vehicle or cockpit, and can catch drowsiness before control is lost. WATCH is image-based.

## 2.2 Eye, mouth and head cues

Eye closure is the most widely used visual cue. PERCLOS, the share of time the eyes are closed over a window, remains the most effective single indicator, and recent work fuses it with blink duration to make it more stable [2]. The eye aspect ratio (EAR) measures eye openness from facial landmarks. It is cheap and needs no training, but a fixed threshold suits some people better than others, which has led to adjusted versions [3]. Yawning is a slower cue, measured by mouth opening, and head pose captures nodding and drooping. Combining the eye aspect ratio, the mouth aspect ratio (MAR) and head pose reaches up to 99% accuracy on NTHU-DDD [4], so systems that combine several cues are more robust than those that rely on one.

## 2.3 Learned classifiers and datasets

Convolutional networks trained on face, eye or mouth images handle lighting, glasses and individual differences better than fixed thresholds. Recent CNNs detect yawning from mouth images [5] and fatigue from both yawning and eye state [6]. The public datasets behind such work include MRL Eye [7], CEW [8], NTHU-DDD [9] and YawDD [10]. On embedded hardware, MobileNetV2, VGG16 and InceptionV3 have been compared for eye-state detection on a Raspberry Pi [11], and a dataset recorded on a Raspberry Pi 5 with Camera Module 3 now covers eye closure and yawning under varied lighting [12]. Body pose estimation has also been studied inside vehicles, where existing models lose accuracy on drivers in real cabins [13].

## 2.4 Gaps addressed by this work

Most systems assume the face is visible. Masks are one exception that recent work has begun to handle [2], but helmets and tinted visors, common for pilots and motorcyclists, hide the eyes as well as the mouth. Many systems also apply the same thresholds to every user. WATCH addresses these gaps: it calibrates to each user, it marks unreadable eyes and mouths as *blocked* instead of guessing, it falls back to body posture when the face is covered, and it runs entirely on a Raspberry Pi 4.

# 3. Data Collection

WATCH was trained on four public datasets, refined with recordings made on the target device, and tested on further data made for this work. Table 1 summarizes all sources, and Figure 1 shows samples from the public ones. Two further sets were made for this work: photos of motorcycle riders in helmets, used only for testing, and YawDD videos with synthetic visors and masks (Section 3.3). The `fetch-data` command downloads all four automatically: MRL Eye, NTHU-DDD and YawDD from Kaggle mirrors, and CEW from the link on its official page.

| Dataset | Content | Subjects | Labels | Used for |
|---|---|---|---|---|
| MRL Eye [7] | 84,898 infrared eye crops, about 80×80 px | 37 | open, closed | eye model |
| CEW [8] | 4,846 eye patches from face photos, 24×24 px | 2,423 | open, closed | eye model |
| NTHU-DDD [9] | 18,000 video frames of drivers, with and without glasses | 4 | drowsy or not, per clip | eye model |
| YawDD [10] | 319 in-car videos from a camera under the mirror | 90 | normal, talking, yawning, per video | yawn and eye models |
| Motorcycle helmet photos (ours) | 6,555 photos of riders in helmets, visors and masks | 11 | alert or drowsy, per session | testing only |
| Synthetic covers (ours) | 90 videos: 30 YawDD videos, each with a dark visor, mirror visor, or visor and mask | 11 | same as YawDD | *blocked* training crops, testing |
Table*: Table 1. Datasets used in this work. The last two were made for this work.

![Figure 1. Samples from the public datasets. Top: full frames from NTHU-DDD and YawDD. Middle and bottom: eye crops from MRL Eye and CEW.](docs/paper/figures/datasets.png){15}

## 3.1 Eye datasets

MRL Eye is the largest source. Its infrared images resemble what the NoIR camera on the Raspberry Pi sees, and they include glasses, reflections and varied lighting. CEW adds variety rather than volume. Its 2,423 subjects were photographed in everyday settings, at much lower resolution. NTHU-DDD contributes video of drivers simulating drowsiness: slow blinks, nodding and yawning, recorded in grayscale and infrared. Its drowsy label applies to a whole clip, not to each frame, so the eye state of every frame had to be derived (Section 4).

## 3.2 Yawning dataset

YawDD is the only source for the yawn model. Each of its 90 drivers was filmed sitting normally, talking and yawning, with and without glasses. Talking matters because it opens the mouth without a yawn, so the model must learn the difference. YawDD also supplies eye images for the eye model. Its 36 videos of 13 drivers wearing sunglasses provide real examples of eyes that cannot be read.

![Figure 2. Frames from YawDD videos. Each driver was filmed sitting normally, talking and yawning, without glasses, with glasses or with sunglasses.](docs/paper/figures/yawdd.png){8.4}

## 3.3 Helmet and covered-face data

WATCH is meant for pilots and riders, whose helmets, visors and masks hide the face. No public dataset covers these conditions, so two further sources were added.

The first is a set of 6,555 photos of 11 motorcycle riders, each recorded once alert and once drowsy, looking forward or away. The helmets are open-face and full-face, with clear or tinted visors, raised or down. Some riders wear glasses, one wears a neck gaiter over the nose and mouth, and one a face mask. The labels were checked by hand. One session filed as both alert and drowsy was removed, and three others were flagged as doubtful. Figure 3 shows six of the riders.

![Figure 3. Photos from the motorcycle helmet dataset: six riders, each in their alert session (top) and drowsy session (bottom).](docs/paper/figures/helmet.png){15}

The second is synthetic. Thirty clear YawDD videos of 11 drivers were overlaid with a dark visor, a mirror visor, and a dark visor with a mask, giving 90 videos in which the true eye and mouth states are still known (Figure 4). Crops from these videos also teach the models what a covered eye or mouth looks like (Section 4.2), and the full videos are replayed through the live monitor for testing.

![Figure 4. Synthetic face covers on a YawDD video, used to test WATCH when the face is hidden.](docs/paper/figures/overlays.png){14}

## 3.4 Device recordings

Public datasets differ from the deployed setting in camera, lighting and angle. To close this gap, WATCH includes a `record` command that collects labelled eye images on the device itself. The buzzer cues the user: one beep to open the eyes, two to close them. A session alternates 12 s open and 6 s closed, three times, so the user never has to look at a screen. The first second after each cue is skipped while the eyes respond. Each eye is cropped and saved with its subject and take name, so complete takes, such as one recorded through sunglasses, can be held out for testing. In total, ==[number]== takes were recorded from ==[number]== people, giving ==[number]== eye crops. These recordings are used only to fine-tune the eye model (Section 6).

# 4. Data Preparation

!>[Figure 5. Offline pipeline, from the WATCH architecture flowchart.](docs/paper/figures/flow_offline.png){7}

Figure 5 shows the offline pipeline. The `prepare-data` command turns the raw sources into labelled eye and mouth crops, adds examples of eyes and mouths that cannot be seen, and splits everything by person. The models then train on these crops (Sections 5 and 6).

## 4.1 Eye and mouth crops

MediaPipe Face Mesh locates facial landmarks in each frame. Eight landmarks around each eye and eight around the mouth define the crop, padded by 15% for the eyes, which are cropped square, and by 30% for the mouth. Crops tilted more than 15° are rotated level, and crops that are almost uniform are discarded. The live monitor uses the same code, so the models see the same kind of crop in training and in use. Every crop is resized to 64×64 px. The eye model sees it in grayscale and the yawn model in colour.

NTHU-DDD frames carry coloured boxes drawn by the dataset's authors, which are removed by inpainting before the landmarks are found. Because NTHU-DDD labels whole clips, each eye crop is labelled by its own eye aspect ratio (EAR): open at 0.35 or above, closed at 0.10 or below. Crops in between are discarded as uncertain. Every third frame is used, up to 150 per clip, which gives 1,333 eye crops from 18,000 frames.

For the yawn model, every tenth frame of the normal and talking videos and every third frame of the yawning videos is cropped, up to 40 per video. Normal and talking frames are labelled *no yawn*. A yawning video also shows the moments before and after the yawn, so only frames with a mouth aspect ratio (MAR) of at least 0.6 are labelled *yawn*.

The clear YawDD videos also supply eye crops. An eye is labelled only when the eye network and the EAR agree: open when the network is at least 95% sure and the EAR is close to the driver's usual value, closed when the network is at least 95% sure and the EAR is well below it. This gives 16,946 further eye crops.

!<

## 4.2 Eyes and mouths that cannot be seen

Each model has a third class, *blocked*, for crops that show no usable eye or mouth (Figure 6). Its examples come from three kinds of source:

- **Real occlusions:** 7,655 eye crops from the YawDD sunglasses videos.
- **Synthetic covers on video:** 4,580 eye crops and 1,377 mouth crops from the visor and mask videos of Section 3.3.
- **Altered crops:** 15,780 MRL Eye and CEW crops darkened by a tint or washed out by a reflection, with glare bands and noise. For the mouth, 3,600 crops covered by a drawn mask, hand or shadow, and 3,395 covered with real content from the same frame: cheek skin shaped as a hand, the shirt below the chin, the background beside the face, or a patch from another video.

In total, 28,015 eye crops and 8,372 mouth crops are labelled *blocked*. Each altered crop stays with the person it came from, so it falls in that person's split.

![Figure 6. Examples of prepared crops, which are resized to 64×64 px for the models. Eyes (top) are open, closed or blocked; mouths (bottom) show no yawn, a yawn, or a cover.](docs/paper/figures/crops.png){14}

## 4.3 Splitting by person

Frames of the same person look alike, so a model tested on someone it trained on scores higher than it would on a stranger. WATCH therefore assigns whole people to the training, validation and test splits, in a 70/20/10 ratio with a fixed seed (42). The assignment is stratified by dataset and by each person's most common label, so every split has a similar mix.

Some groups are too small to leave to chance. NTHU-DDD has only four people, and only 13 YawDD drivers wear sunglasses, so seven people are pinned to a split: NTHU-DDD subject 002 and YawDD drivers female17, female22 and male38 to test, and NTHU-DDD subject 006 and YawDD drivers female18 and male36 to validation. The yawn split keeps the assignment it had before the *blocked* class was added, so earlier yawn results stay comparable. A final check rejects the split if any person appears in more than one part. Table 2 gives the result.

| Crops | Train | Validation | Test | Total |
|---|---|---|---|---|
| Eye model: people | 1,784 | 512 | 258 | 2,554 |
| Eyes closed | 33,336 | 8,289 | 3,980 | 45,605 |
| Eyes open | 44,905 | 11,028 | 6,485 | 62,418 |
| Eyes blocked | 16,008 | 7,334 | 4,673 | 28,015 |
| Yawn model: people | 63 | 18 | 9 | 90 |
| No yawn | 5,578 | 1,680 | 914 | 8,172 |
| Yawn | 1,510 | 335 | 258 | 2,103 |
| Mouth blocked | 5,537 | 1,877 | 958 | 8,372 |
Table: Table 2. Crops per split. Eye people include every CEW face, since CEW does not identify its subjects.

# 5. Model Design

WATCH uses two small convolutional networks, one for the eyes and one for the mouth. Both take a 64×64 crop and end in a softmax over three classes. They are small on purpose: the whole system, including face and body tracking, has to run on the CPU of a Raspberry Pi 4. Figure 7 shows both networks and Table 3 summarizes them.

![Figure 7. The two networks. The yawn model combines an unchanged yawn branch with a separate branch that detects a covered mouth.](docs/paper/figures/models.png){16.5}

## 5.1 Eye model

Three convolution blocks, each a 3×3 convolution (32, 64 and 128 filters) with ReLU, batch normalization and 2×2 max pooling, reduce the crop to an 8×8×128 feature map. Global average pooling, dropout of 0.5, a dense layer of 128 units and a second dropout of 0.5 lead to the softmax over *closed*, *open* and *blocked*. The crop is converted to grayscale and copied to three channels, so infrared images, like those from MRL Eye and the NoIR camera, and colour webcam images look alike to the model.

## 5.2 Yawn model

The yawn network is smaller still: two convolution blocks of 16 and 32 filters without batch normalization, global average pooling and a dense layer of 64 units. It keeps the crop in colour.

The *blocked* class was added after the yawn model was trained. Retraining the whole network with three classes lowered its accuracy on mouths that were not yawning, so the final model is built from two branches of the same architecture instead. The yawn branch is the original two-class model, unchanged. The blocked branch was trained on all three classes, and only its probability of *blocked*, *p~b~*, is used. Two fixed layers combine them into [(1 − *p~b~*)·P(no yawn), (1 − *p~b~*)·P(yawn), *p~b~*]. When the mouth is visible, *p~b~* is close to zero and the yawn decision is the same as the original model's.

## 5.3 From probabilities to a decision

*Blocked* wins only when its probability reaches a threshold: 0.9 for the eyes and 0.7 for the mouth. Otherwise it is set aside, the other two probabilities are rescaled to sum to one, and the larger one decides. The eye threshold is high because calling a visible eye *blocked* would hide a real eye closure. The mouth threshold is lower because a mouth covered by a hand or a mask is otherwise easily read as a yawn.

| Model | Input | Convolution blocks | Classes | Parameters | TFLite size |
|---|---|---|---|---|---|
| Eye | 64×64, grayscale | 32, 64, 128 filters, with batch normalization | closed, open, blocked | 111,043 | 439 KB |
| Yawn | 64×64, colour | two branches of 16, 32 filters | no yawn, yawn, blocked | 14,737 | 65 KB |
Table*: Table 3. The two networks. TFLite files are exported for embedded use; the Raspberry Pi currently runs the Keras models (Section 8.2).

# 6. Training

## 6.1 Settings

Both models are trained with the Adam optimizer on categorical cross-entropy. Class weights, set inversely to each class's share of the training crops, keep the smaller classes such as *yawn* from being ignored. Training crops are flipped horizontally, rotated by up to 10°, zoomed and shifted by up to 10%, and brightened or darkened by up to 20%. Validation and test crops are left unchanged. Training stops once the validation loss has not improved for several epochs, and the weights from the best epoch are kept. The learning rate is cut to 30% whenever the validation loss stalls. Table 4 lists the values, all with the seed fixed at 42.

| Setting | Eye model | Yawn and blocked branches |
|---|---|---|
| Learning rate | 0.0001 | 0.001 |
| Batch size | 64 | 32 |
| Maximum epochs | 30 | 80 |
| Early stopping | after 5 epochs | after 8 epochs |
| Learning-rate cut | after 2 epochs | after 3 epochs |
| Dropout | 0.5 | 0.5 |
Table: Table 4. Training settings. Early stopping and learning-rate cuts follow the validation loss.

Each run produces a candidate model, which replaces the saved model only if its validation accuracy is higher. A run that is worse is kept aside rather than lost.

## 6.2 Training the yawn model in two stages

The yawn branch is trained first, as a two-class model on the *no yawn* and *yawn* crops. The blocked branch is then trained separately on all three classes with the same settings. The two are joined by the fixed layers of Section 5.2 without further training, so adding the blocked branch cannot change how the yawn branch reads a visible mouth.

## 6.3 Training curves

Figure 8 shows a training run of each part, repeated for this paper with the same settings and seed; the saved models were not replaced. On a laptop RTX 4050 GPU the eye model took 5.3 minutes and stopped after 15 epochs, keeping epoch 10. The yawn and blocked branches took under a minute each and kept epochs 8 and 10. The eye model's validation loss jumps at epochs 4 and 5 and settles once the learning rate is cut. Its validation accuracy stays about five points below training accuracy, because validation holds people the model has never seen. Training on a GPU is not exactly repeatable, so these runs differ slightly from the saved models. Section 9 evaluates the saved models.

## 6.4 Fine-tuning on the device

Public datasets differ from the camera, light and angle of the deployed device, so the eye model can be fine-tuned on the recordings of Section 3.4 with the `finetune` command. Only the dense layers learn. The convolution layers stay fixed, because a few takes are too narrow to retrain them without losing what they learned from the public data. Whole takes are held out for testing, either a quarter of them at random or named takes such as one recorded through sunglasses, and 15% of the remaining crops are used for validation. Fine-tuning uses a learning rate of 0.00005 for up to 15 epochs, always starts from the original model, and keeps the result only if it scores higher on the held-out takes.

![Figure 8. Training curves of the eye model and the two yawn branches. The dashed line marks the epoch whose weights are kept; dotted lines mark learning-rate cuts.](docs/paper/figures/training_curves.png){15.5}

# 7. Real-Time Alertness System

The live monitor turns each camera frame into a level of alertness: normal, warning or critical. Figure 9 shows the steps for one frame. Every threshold below comes from one configuration file, which the Raspberry Pi monitor and the browser demo share.

![Figure 9. Processing of one frame in the live monitor, condensed from the WATCH architecture flowchart (Appendix).](docs/paper/figures/live_pipeline.png){16.5}

## 7.1 Calibration

Each session starts with 4 s of calibration while the user looks at the camera normally. WATCH records the eye aspect ratio of both eyes, the head's pitch and roll, and the sitting posture: the height, width and tilt of the shoulder line and the height of the head above it. The user's resting EAR becomes their *open* level and 0.10 below it their *closed* level. If fewer than ten readings were taken, or the resting EAR is under 0.20, as behind sunglasses, the default levels of 0.35 and 0.10 are used instead. The head counts as drooped 25° past its resting pitch or roll. Calibration runs again when the user comes back after 3 s out of view, and when a lost camera reconnects.

## 7.2 Processing each frame

Bright highlights on the face, such as visor reflections, are removed by inpainting, and frames darker than 30% of full brightness are brightened. MediaPipe Face Mesh then finds the face. If it fails, a face detector locates the face and the mesh is run again on an enlarged crop. MediaPipe Pose runs every third frame, because posture changes slowly and the model costs about as much as the face tracking.

## 7.3 Cue scores

Each cue becomes a score from 0 to 1, smoothed by a rolling median over five frames.

- **Eyes.** The eye network gives the probability that each eye is closed. An eye counts as unreadable when the network says *blocked* or is less than 65% confident. The two eyes are averaged, weighted by their width, so the eye nearer the camera counts more.
- **Mouth.** The yawn network gives the probability of a yawn. The mouth counts as unreadable when the network says *blocked*, is less than 65% confident, or reports a yawn while the mouth aspect ratio shows a closed mouth. Otherwise the score is the larger of the network's probability and the MAR scaled from 0.3 (closed) to 0.6 (open).
- **Head.** Pitch and roll past the calibrated resting pose are each scaled from 0 at rest to 1 at the drooped angle and combined as a vector, so a diagonal droop counts as fully as a straight one.
- **Body.** The shoulder line is compared with the calibrated posture, measured in shoulder widths so the distance to the camera does not matter. The body is slumped when the shoulders drop by 0.10 widths, tilt 10° further, or look 20% wider from leaning toward the camera, or when the head drops 0.25 widths toward them. While a hand is raised to the face or helmet, the tilt is ignored, because lifting an arm tilts the shoulders.

## 7.4 Signs of drowsiness

A cue becomes a *sign* when its score stays high for a set time, or when its average over a longer window is high, in the manner of PERCLOS [2]. For yawns and slumps, dips of up to 0.5 s are forgiven, so a moment of closing the mouth does not restart a yawn. Three signs come from the body's movement rather than a score: nodding off, a long stillness, and an unanswered check-in. Table 5 lists them.

| Sign | Sustained | Or average over a window |
|---|---|---|
| Eyes closed | score ≥ 0.85 for 1.5 s | ≥ 0.75 over 5 s |
| Yawning | score ≥ 0.5 for 2 s | ≥ 0.8 over 8 s |
| Head drooped | fully drooped for 1.5 s | ≥ 0.9 over 6 s |
| Slumped | fully slumped for 2 s | ≥ 0.9 over 6 s |
| Nodding off | 3 nods within 60 s: the head drops 0.12 widths and returns within 2.5 s | |
| No movement | shoulders and head move less than 0.05 widths in 45 s | |
| No check-in answer | no answer within 5 s of a check-in | |
Table: Table 5. Signs of drowsiness and their limits.

## 7.5 Escalation

Signs drive a three-level state machine (Figure 10). The first sign raises a *warning* and starts a 30 s watch. The level becomes *critical*, and the alarm sounds, if a second kind of sign appears during the watch, if the same sign repeats, or if it lasts: eyes closed for 3 s, head drooped or body slumped for 4 s, or no movement for 30 s. Eyes, head, body, nodding and stillness must repeat twice, but yawns three times, because a single yawn is common in alert people. Critical drops back to a warning after 5 s without signs, and a warning ends when its watch runs out with no signs.

![Figure 10. Alert escalation. While nobody is in view the level and the watch are paused; if nobody returns within 5 s the alert is cleared.](docs/paper/figures/escalation.png){8}

The level is held while the user is out of view, so looking away cannot clear an alert, but it is cleared after 5 s with nobody in view, since no one is left to respond. If the camera disconnects, the dashboard shows CAMERA LOST, the alarm sounds after 3 s, and the monitor recalibrates once the camera returns.

## 7.6 Covered faces and check-ins

WATCH has three modes. In the default *both* mode, the face and body cues count together, so the monitor works with or without a helmet. *Face* mode reads the face first and counts the body only when the face cannot be read. *Body* mode skips the face models entirely, which suits a user whose helmet always hides the face and saves processing on the Pi.

When the face is not found but the shoulders are, the dashboard shows FACE COVERED and monitoring continues on posture, nodding and stillness. Body language cannot tell whether eyes behind a tinted visor are open, though. So when the eyes have been unreadable for 60 s while the user is still in view, the monitor beeps and asks for a check-in: a press of the space bar, or the "I'm awake" button in the browser. No answer within 5 s raises a warning, and 5 s more makes it critical. Each answer and its reaction time are written to the session log. Figure 11 shows both situations.

![Figure 11. The live monitor on the Raspberry Pi with a pilot whose face is visible (left) and with a tinted visor and oxygen mask (right).](docs/paper/figures/pilot_placeholder.png){12.5}

# 8. Implementation and Deployment

## 8.1 Software

WATCH is written in Python 3.12 with TensorFlow, MediaPipe and OpenCV. Every step is an action of one command-line tool, `python -m watch`, from `fetch-data`, `prepare-data`, `train` and `evaluate` to `export`, `live`, `record` and `finetune`, and a `pipeline` action reruns the public-data steps of Sections 3 to 6 in order. The *blocked* examples and the joining of the two yawn branches are produced by separate experiment scripts. Setup scripts install a fixed Python and pinned packages, and 86 unit tests cover the data split, the models, the alert logic, the camera stream and the reports. Training runs on a Linux machine with a GPU; the live monitor runs on Windows, Linux and the Raspberry Pi. The `export` action writes the models in three formats, Keras, TensorFlow.js for the browser and TFLite for embedded use, together with a JavaScript copy of the configuration, so the browser applies the same thresholds as the Pi.

## 8.2 Raspberry Pi

The embedded monitor runs on a Raspberry Pi 4 Model B with 8 GB of memory, a 32 GB microSD card and a Camera Module 3 NoIR (Figure 12, Table 6). The camera has no infrared filter, so it can also see by infrared light. The Pi has no screen. Instead, the monitor streams its window, the video with the dashboard beside it, to a browser on a laptop over Wi-Fi or a direct Ethernet cable. The page has buttons to mute the alarm, export a report, stop the monitor and shut the Pi down. A buzzer on a GPIO pin sounds the alarm, and the page can beep as well. A system service starts the monitor at boot and restarts it if it stops.

![Figure 12. The Raspberry Pi setup. The Pi needs no screen; a laptop on the same network views and controls the monitor.](docs/paper/figures/pi_hardware.png){8}

| Part | Details |
|---|---|
| Computer | Raspberry Pi 4 Model B, 8 GB, Raspberry Pi OS (Debian 13) |
| Storage | 32 GB microSD |
| Camera | Camera Module 3 NoIR, read at 640×480 |
| Alarm | buzzer module on GPIO 4 |
| Power | 5 V 3 A USB-C supply |
| Cooling | heatsink; the Pi reached about 71 °C under load |
| Link to the laptop | Wi-Fi or Ethernet, port 8766 |
Table: Table 6. Hardware of the embedded monitor.

The developer measured about 4 to 6 frames per second in the default *both* mode with a face in view, and about 8 to 10 in *body* mode, which skips the face models. Because the signs of Table 5 are timed in seconds, the alert logic holds at these rates, but eye closures shorter than a few frames can be missed. Pose tracking on every frame overloaded the Pi until it froze, so it runs every third frame, or every second frame in body mode. The Pi still runs the Keras models; switching it to the exported TFLite files should be faster but has not yet been done.

![Figure 13. The Raspberry Pi setup (left) and a session report exported on the Pi (right).](docs/paper/figures/pi_placeholder.png){12.5}

## 8.3 Browser demo

The `web` action serves a browser version of the monitor. It runs entirely in the browser with TensorFlow.js and MediaPipe's face and pose landmarkers, so no video leaves the computer. It follows the face flow of Section 7 with the slump sign and the check-in, but not the other body-language signs. It also analyses a single uploaded or pasted photo (Figure 14).

![Figure 14. The browser demo analysing a photo from YawDD. The scanned view marks the eyes, mouth and head with their readings: both eyes open, a yawn, and the head upright.](docs/paper/figures/web_demo.png){14}

## 8.4 Dashboard and session reports

Beside the video, the dashboard shows the alert level, each sign with its timer, what is being detected, such as how many eyes are readable, the lighting and the sharpness of the face, and the recent level changes. Exporting a session saves two files: a graph of the alert level over time, with periods of no face, a covered face, calibration and a lost camera shaded, and a CSV log of every change with its reason. Check-in answers and their reaction times are part of the log.

# 9. Results

## 9.1 Accuracy on unseen people

Both models were tested on the test split of Section 4.3, whose people the models never saw in training, using the decision rule of Section 5.3. Table 7 gives the results and Figure 15 the confusion matrices.

| Measure | Eye model | Yawn model |
|---|---|---|
| Test people | 258 | 9 |
| Test crops | 15,138 | 2,130 |
| Accuracy, three classes | 96.5% | 95.0% |
| Accuracy on visible eyes or mouths | 96.7% | 93.4% |
| Visible ones wrongly marked *blocked* | 1.2% | 0.9% |
| *Blocked* ones caught | 96.1% | 96.9% |
| ROC-AUC, closed eye or yawn | 0.994 | 0.990 |
Table: Table 7. Test results on people not seen in training.

The eye model reads 96.3% of closed eyes and 97.0% of open eyes correctly, and marks only 1.2% of visible eyes as *blocked*, so a covered eye rarely hides a real closure. It is strongest on MRL Eye, at 97.4% for closed and 95.6% for open eyes, and weakest on closed eyes in the small, low-resolution CEW patches and on the four-person NTHU-DDD set, at 87 to 89%. The yawn model is the weaker of the two. It misses 17.4% of yawns, reading them as an ordinary mouth, while only 2.3% of non-yawning mouths are taken for a yawn. Its test split holds only nine people, so its figures are less certain than the eye model's.

![Figure 15. Confusion matrices on the test split. Each row shows how the crops of one actual class were classified.](docs/paper/figures/confusion.png){13}

## 9.2 Speed

On the development laptop with an RTX 4050 GPU, one prediction took 11.5 ± 2.0 ms for the eye model and 11.0 ± 1.7 ms for the yawn model, averaged over 100 runs. These times cover the networks alone. On the Raspberry Pi the complete monitor, including face and body tracking, ran at 4 to 6 frames per second in *both* mode and 8 to 10 in *body* mode (Section 8.2).

## 9.3 Motorcycle helmet photos

The helmet photos of Section 3.3 test WATCH on real riders, with the duplicated session removed: 5,954 photos of 10 riders. The photos are single frames, not video, so they test the per-frame readings rather than the alert timing. Table 8 compares the alert and drowsy sessions.

| Reading | Alert photos | Drowsy photos |
|---|---|---|
| Face found | 83.6% | 85.1% |
| Eye readings: closed | 3.7% | 38.2% |
| Eye readings: *blocked* | 8.6% | 3.5% |
| Mouth readings: *blocked* | 45.0% | 62.1% |
| Mouth readings: yawn | 31.6% | 12.6% |
| Body pose found | 69.8% | 55.9% |
Table: Table 8. Readings on the motorcycle helmet photos.

The eye model separates the sessions clearly: closed-eye readings rise from 3.7% when alert to 38.2% when drowsy. Every rider whose face was found in both sessions has more closed-eye readings in the drowsy one, including all seven riders with trusted labels (Figure 16). The face was found in 84 to 85% of photos, but much less often at side angles: never in P10's alert session, and in 46% of P07's drowsy session.

![Figure 16. Share of eye readings classed closed for each rider. Asterisks mark doubtful labels. P10's face was not found in the alert session.](docs/paper/figures/helmet_closed.png){8}

The mouth results are mixed. Helmets and face covers hide the mouth in about half the photos, and the *blocked* class catches most of them. Two covers fool it, however: P02's neck gaiter is read as a yawn in 90% of alert photos and the vents in P05's chin bar in all of them. Because such a cover stays in place, a live session would likely see these readings persist and raise a false yawn warning. Body pose was found in 70% of alert and 56% of drowsy photos, but in only 4 to 23% for P03, P04 and P11, whose photos are close-up and leave the shoulders out of frame.

# 10. Limitations and Future Work

## 10.1 Limitations

**Data.** Some groups are small. NTHU-DDD has four people, the yawn model is tested on only nine, and the helmet set has eleven riders recorded as photos with one label per session. Part of the labels were derived automatically, from the eye aspect ratio in NTHU-DDD and from agreement between the eye network and the EAR in YawDD, so some crops may be mislabelled. No public data shows pilots in a cockpit, so visors and oxygen masks were tested on synthetic covers and on riders, not on pilots.

**Evaluation.** The models were tested crop by crop and the helmet photos frame by frame. The complete alert logic, with its timings and escalation, has not yet been measured against an independent measure of drowsiness, such as a reaction-time test, on continuous video. The tool that replays videos through the monitor no longer matches the live code and has to be updated first.

**Models.** The yawn model misses about one yawn in six, and some covers fool it: a neck gaiter and the vents in a chin bar were read as yawns. Fine-tuning on device recordings teaches only open and closed eyes, so repeated fine-tuning may weaken the *blocked* class.

**On the device.** The body-language limits are starting values that still need tuning with people in real helmets and seats, and a real nod under a mask has not yet been confirmed to raise an alert. In *both* mode the Pi runs at 4 to 6 frames per second, so very short eye closures can be missed. The face is often lost at side angles, and the body tracker needs the head and both shoulders in view. Reading eyes through a tinted visor with the NoIR camera, use in full darkness, and the infrared lights have not yet been tested. The browser page has no login, so it should only be used on a trusted network.

## 10.2 Future work

The most important next step is a study with pilots or riders wearing their own equipment, recorded as continuous video with per-frame labels and a reference measure of drowsiness. That would allow the body limits to be tuned, the full alert logic to be evaluated, and the replay tool to be brought up to date for repeatable tests. The yawn model needs training data with neck gaiters, chin bars and infrared light, and fine-tuning should keep the *blocked* class. On the device, running the exported TFLite models should raise the frame rate, and the infrared lights would make the camera usable at night.

# 11. Conclusion

WATCH monitors alertness from a single camera by combining two small networks with face-landmark geometry, head pose and upper-body posture, calibrated to each user and escalated over time from warning to critical. A *blocked* class lets each network say when it cannot see, instead of guessing, and the monitor falls back on posture and check-ins when a visor or mask hides the face. On people not seen in training, the eye model classified 96.5% of crops correctly and the yawn model 95.0%. On photos of motorcycle riders in helmets, closed-eye readings rose from 3.7% in alert sessions to 38.2% in drowsy ones. The whole system runs on a Raspberry Pi 4 at 4 to 6 frames per second. The main open steps are a study on continuous video of pilots or riders, tuning of the body limits, and a yawn model that is not fooled by face covers.

\newpage

# References

[1] Y. Albadawi, M. Takruri and M. Awad, "A review of recent developments in driver drowsiness detection systems," *Sensors*, vol. 22, no. 5, 2069, 2022.

[2] Y. Ling and X. Weng, "Efficient and robust driver fatigue detection framework based on the visual analysis of eye states," *Promet - Traffic & Transportation*, vol. 35, no. 4, pp. 567-582, 2023.

[3] C. Dewi, R.-C. Chen, X. Jiang and H. Yu, "Adjusting eye aspect ratio for strong eye blink detection based on facial landmarks," *PeerJ Computer Science*, vol. 8, e943, 2022.

[4] Y. Albadawi, A. AlRedhaei and M. Takruri, "Real-time machine learning-based driver drowsiness detection using visual features," *Journal of Imaging*, vol. 9, no. 5, 91, 2023.

[5] F. Majeed, U. Shafique, M. Safran, S. Alfarhood and I. Ashraf, "Detection of drowsiness among drivers using novel deep convolutional neural network model," *Sensors*, vol. 23, no. 21, 8741, 2023.

[6] F. Makhmudov, D. Turimov, M. Xamidov, F. Nazarov and Y.-I. Cho, "Real-time fatigue detection algorithms using machine learning for yawning and eye state," *Sensors*, vol. 24, no. 23, 7810, 2024.

[7] "MRL Eye Dataset," Kaggle, https://www.kaggle.com/datasets/tauilabdelilah/mrl-eye-dataset, accessed Nov. 2026.

[8] "Closed Eyes in the Wild (CEW)," Nanjing University of Aeronautics and Astronautics, https://parnec.nuaa.edu.cn/_upload/tpl/02/db/731/template731/pages/xtan/ClosedEyeDatabases.html, accessed Nov. 2026.

[9] "NTHU-DDD," Kaggle, https://www.kaggle.com/datasets/ikhlaselhamly/nthu-ddd, accessed Nov. 2026.

[10] "YawDD dataset," Kaggle, https://www.kaggle.com/datasets/enider/yawdd-dataset, accessed Nov. 2026.

[11] A. K. Pathak, A. K. Singh, P. Kumar, V. Bhatia and O. Krejcar, "Real-time anti-sleep alert algorithm to prevent road accidents to ensure road safety," *Frontiers in Future Transportation*, vol. 6, 1545411, 2025.

[12] S. Liawatimena and N. Isworo, "Annotated drowsiness detection dataset captured using Raspberry Pi 5," *Data in Brief*, vol. 63, 112211, 2025.

[13] R. Guesdon, C. Crispim-Junior and L. Tougne, "DriPE: A dataset for human pose estimation in real-world driving settings," in *Proc. IEEE/CVF ICCV Workshops*, pp. 2865-2874, 2021.


\newpage

# Appendix A. Architecture flowchart

Figures A1 to A3 show the full WATCH architecture flowchart in three parts, from offline training to the live monitor's outputs and the photo analysis.

![Figure A1. Architecture flowchart, part 1: offline model training, start-up and per-frame input and detection.](docs/paper/figures/flowchart_part1.png){14.8}

\newpage

![Figure A2. Architecture flowchart, part 2: calibration, feature extraction, sign detection and the start of alert escalation.](docs/paper/figures/flowchart_part2.png){16.5}

\newpage

![Figure A3. Architecture flowchart, part 3: alert escalation, outputs, user controls and photo analysis.](docs/paper/figures/flowchart_part3.png){16.5}

\newpage

# Appendix B. Configuration

All values live in `configs/config.yaml`. Table B1 lists those not already given in Tables 4 and 5.

| Setting | Value |
|---|---|
| Input size | 64×64 px |
| Split ratios, seed | 70/20/10, seed 42 |
| *Blocked* thresholds, eye and yawn | 0.9 and 0.7 |
| Minimum network confidence | 0.65 |
| Smoothing window | 5 frames |
| Default EAR, open and closed | 0.35 and 0.10 |
| Default MAR, closed and open | 0.3 and 0.6 |
| Calibration time | 4 s |
| Personal closed EAR | resting EAR − 0.10, if resting EAR ≥ 0.20 |
| Drooped head | resting pitch or roll + 25° |
| Escalation watch, calm time | 30 s and 5 s |
| Body slump limits | drop 0.10, tilt 10°, head drop 0.25, forward 0.20 |
| Nodding off | 3 nods of 0.12 widths within 60 s, each back within 2.5 s |
| No movement | under 0.05 widths for 45 s |
| Check-in | after 60 s of unreadable eyes, answer within 5 s |
| Pose tracking | every 3rd frame, every 2nd in body mode |
| Low light | frames under 30% brightness raised to 45% |
| Recalibration | after 3 s out of view |
| Alert cleared | after 5 s with nobody in view |
| Camera lost | retry every 1 s, alarm after 3 s |
| Buzzer | GPIO 4 |
Table: Table B1. Further configuration values.

# Appendix C. Reproducing the results

On a Linux machine with a GPU, `./scripts/setup.sh` installs the environment. `./scripts/pipeline.sh` then downloads the public datasets, prepares the crops and splits, trains both models, evaluates them on the test split and exports them. The live monitor runs with `./scripts/live.sh`, or `./scripts/live.sh --camera pi --stream` on the Raspberry Pi after `./scripts/setup-pi.sh`. The *blocked* examples, the joined yawn model and the helmet measurements come from the scripts in `experiments/hidden_face/`.
