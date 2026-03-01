# AI-AGENT-IN-BUILDINGS
AI agents for smart energy management in a university
-------------------------------------------
Project Overview

This project focuses on designing and developing an intelligent room monitoring and automation system capable of processing environmental data and generating logical automated responses. The system analyzes temperature and humidity readings from different rooms and makes decisions based on predefined workflow rules. At the current stage, the system operates using manually entered datasets. The main objective is to build a stable and logically consistent workflow prototype before integrating real-time hardware components such as sensors.
------------------------------------------
Dataset Description

Currently, the project includes environmental datasets collected from three different rooms: Mailroom, Lab, and Kitchen. For each room, two types of data are included: temperature data and humidity data. These datasets are stored in CSV format and are used for preprocessing, validation, and visualization.
------------------------------------------
Preprocessing and Analysis

The Temperature_and_Humidity_Preprocessing.py script performs data cleaning and preprocessing, handles missing or invalid values, converts data types where necessary, and generates basic visualizations of environmental trends. The goal is to ensure that the system can correctly read, process, and interpret environmental parameters before automation logic is applied, ensuring accuracy and stability in the workflow.
------------------------------------------
Current Development Stage

The system is currently in the prototype phase and relies on manual dataset input. Workflow logic testing, data preprocessing, validation, and visualization of environmental trends have been implemented. The system is being tested to ensure it can handle different input scenarios consistently without errors.
------------------------------------------
Future Work

Future development will focus on integrating real-time sensors, automating data collection, deploying intelligent decision-making workflows, and expanding the system to include additional environmental parameters for a more comprehensive monitoring solution.
------------------------------------------
