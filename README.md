# AI-AGENT-IN-BUILDINGS
AI agents for smart energy management in a university
-------------------------------------------
Project Overview:

-This project focuses on designing and developing an intelligent room monitoring and automation system capable of processing environmental data and generating logical automated responses. 
The system analyzes temperature and humidity readings from different rooms and makes decisions based on predefined workflow rules.
At the current stage, the system operates using manually entered datasets. The main objective is to build a stable and logically consistent workflow prototype before integrating real-time hardware components such as sensors.
------------------------------------------
Dataset Description:

Currently, the project includes environmental datasets collected from three different rooms:
-Mailroom
-Lab
-Kitchen

For each room, two types of data are included:
-Temperature data
-Humidity data

!These datasets are stored in CSV format and are used for preprocessing, validation, and visualization.
------------------------------------------
Preprocessing and Analysis

The Temperature_and_Humidity_Preprocessing.py script performs:
-Data cleaning and preprocessing
-Handling of missing or invalid values
-Data type conversion
-Basic visualization of environmental trends
-Preparation of structured data for decision-making workflows
-The goal is to ensure that the system can correctly read, process, and interpret environmental parameters before automation logic is applied.
------------------------------------------
Current Development Stage

-Manual dataset input
-Workflow logic testing
-Data preprocessing and validation
-Visualization of environmental trends

!The system is currently being tested to ensure it can handle various input scenarios without failure.
------------------------------------------
Future Work

-Integration of real-time sensors
-Automation of data collection
-Deployment of intelligent decision-making workflows
-Expansion to additional environmental parameters
