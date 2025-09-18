# 🤖 Expressive Humanoid Robot Head  

A humanoid robot head that sees, expresses, and speaks emotions in real time.  

A bio-inspired robotic head capable of *head movement, eye expression, and emotion detection* with voice output. This project integrates *computer vision (OpenCV + Python), **Arduino-based servo control, and **LED-based eye expressions*, drawing inspiration from cutting-edge humanoid robot research.  
 
(Block diagram showing: camera input → Python AI → Arduino control → Servos + LEDs)  
---

## 📌 Overview  

Humans communicate not only with words but also through *facial expressions, gestures, and head movement*.  
This project replicates those features in a *humanoid robot head* that can:  

- Move its *head & jaw* (servo-driven)  
- Express emotions via *LED “eyes”*  
- Detect human emotions in real-time (smile, sad, happy, angry)  
- Announce detected emotions through *speech output*  

The goal is to bridge *AI-driven perception* and *robotic actuation* for natural, expressive interactions.  

---

## 🚀 Features  

- 🎥 *Emotion Detection* with OpenCV  
- 🧠 *Python Controller* → serial link to Arduino  
- 🔄 *Servo-driven head & jaw movement*  
- 👀 *LED eyes* simulate eye opening/closing  
- 🔊 *Speech Output* with synchronized jaw motion  

---

## 🛠 Components  

- Arduino UNO × 3  
- Servo Motors (neck + jaw)  
- LEDs (eyes)  
- Camera (PC/Laptop input)  
- LCD Displays × 2  
- Push Buttons × 6  
- DC Pump (optional mechanical demo)  
- Python libraries: opencv-python, pyserial, pyttsx3  

---

## ⚙ Architecture  

1. *Camera captures video stream*  
2. *Python (OpenCV)* → detects facial emotion  
3. *Serial Commands → Arduino*  
   - Moves servos (head, jaw)  
   - Toggles LED eyes  
4. *Speaker output* → announces detected emotion  

---

## 📖 References  

This project is backed by recent academic research:  

- Yan, Z. et al. Facial Expression Realization of Humanoid Robot Head and Strain-Based Anthropomorphic Evaluation of Robot Facial Expressions. Biomimetics 2024, 9(3), 122.  
  👉 [Read Paper](https://www.mdpi.com/2313-7673/9/3/122)  

- Said, S. et al. Design and Implementation of Adam: A Humanoid Robotic Head with Social Interaction Capabilities. Appl. Syst. Innov. 2024.  
  👉 [Read Paper](https://www.researchgate.net/publication/380921246_Design_and_Implementation_of_Adam_A_Humanoid_Robotic_Head_with_Social_Interaction_Capabilities)  

---

## 📚 Related Books  

For deeper insights into robotics, AI, and humanoid systems, here are recommended reads:  

- 🤖 *“Introduction to Autonomous Robots”* – Nikolaus Correll, Bradley Hayes, et al.  
- 🤖 *“Springer Handbook of Robotics”* – Bruno Siciliano, Oussama Khatib  
- 🤖 *“Humanoid Robotics: A Reference”* – Prahlad Vadakkepat, AM Mujtaba  
- 🤖 *“Facial Expression Recognition: From Human to Humanoid Robots”* – Fernando De la Torre  
- 🤖 *“Artificial Intelligence: A Modern Approach”* – Stuart Russell, Peter Norvig  
- 🤖 *“Designing Sociable Robots”* – Cynthia Breazeal  

---

## 🎯 Use Cases  

- 🤝 Human–Robot Interaction Research  
- 🎓 Educational Demonstrations in AI/Robotics  
- 🧪 Emotion Detection & HRI Experiments  
- 🎤 Hackathons & Exhibitions (interactive showcase)  

---

## 🏗 Setup / Installation  

1. Clone the repo:  
   ```bash
   git clone https://github.com/<username>/<repository-name>.git

   cd <repository-name>
