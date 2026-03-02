# 🐟 Big Fish Eats Small Fish

An ecological simulation game driven by Deep Q-Network (DQN) reinforcement learning.
Two independent AI agents — large predatory fish and small herbivorous fish — learn
survival strategies through interaction in a dynamic underwater ecosystem.

---

## 📺 Demo

> **YouTube:** [https://youtu.be/YourVideoLink](https://youtu.be/YourVideoLink)

---

## 🎮 Features

- Two independent DQN agents trained simultaneously (large fish vs small fish)
- Directional field-of-view (FOV) perceptual model
  - Large fish: 120° forward hunting cone
  - Small fish: 140° predator detection cone + 160° foraging cone
- Orientation-aware rendering: directional eyes, fins, and FOV arc overlay
- Gender-based reproduction system with breeding cooldown
- Progressive hunger mechanics — both species can starve to death
- Real-time population trend graphs and reward history in UI
- Headless training mode for up to 8× faster throughput
- Auto-save and manual model checkpointing

---

## 🗂️ Project Structure
