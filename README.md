# Neuromorphic-AI-Robotic-Prosthetic-Hand
This project presents a low cost neuromorphic-controlled prosthetic hand that uses brain-inspired intelligence to help with differently abled people regain the functionality of their natural hands. This focuses on Spiking neural networks (SNNs), Artificial Intelligence models that mimics the brain signals and make a platform to facilitate energy-efficient, real-time motor control and signal processing. The project integrates bio-inspired neuromorphic computing with Arduino-based hardware to achieve real-time, energy-efficient gesture control.
## Objectives
- This project aims to bridge the gap between: Low-cost but limited prosthetics and Advanced but expensive prosthetic systems.
- We aim to develop: human-like gesture simulation, Implement neuromorphic-inspired AI and Improve control accuracy.
- The system provides: Real-time processing, Energy-efficient computation, Natural gesture simulation, Modular and upgradeable architecture.
## technologies Used
- Software: Arduino IDE.
            Python for microcontroller programming.
            Spiking Neural Network (SNN) algorithm implementation.
            Signal preprocessing and noise filtering algorithms.
- Hardware: Arduino Microcontroller.
            Surface EMG (sEMG) Sensors.
            Servo Motors (for fingers and wrist).
            3D Printed Prosthetic Structure.
            Li-ion Battery Pack.
 ## Working Principle
1. Signal Detection: When the user intends to move their hand, muscle signals (EMG signals) or control inputs are generated. These signals are captured using sensors placed on the user’s arm or through an external control interface.
2. Signal Processing: The raw signals collected from the sensors are usually weak and noisy. They are first amplified, filtered, and converted into digital form using a microcontroller. This step ensures that only useful movement-related information is sent for further processing.
3. AI Processing: The processed signals are then fed into a neuromorphic AI model that works similar to human neurons.This model analyzes the patterns in the signals and identifies the intended gesture, such as gripping, pointing, or opening the hand. Over time, the system improves its accuracy by learning from repeated movements and user feedback.
4. Decision and Control: After recognizing the intended gesture, the AI system generates appropriate control commands. These commands are sent to the motor control unit, which decides how much each finger should move and in which direction.
5. Actuation and Movement: Servo motors or actuators connected to the fingers receive the control signals. They rotate or move accordingly, allowing the prosthetic hand to perform human-like gestures and gripping actions.
  
