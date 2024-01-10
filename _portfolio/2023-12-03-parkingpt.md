---
title: "ParkinGPT"
excerpt: "A GPT-drived interactive app for greener, safer, and smarter parking<br/><img src='/images/parkingpt.jpeg'><br/>Image Source: [ChatGPT](https://chat.openai.com/)"
collection: portfolio
portfoliolink: "https://devpost.com/software/destchat"
date: 2023-12-03
---

## Inspiration

Parking, both essential and frustrating for drivers, is a daily topic in our lives that we cannot avoid. Have you ever experienced a sense of despair while navigating through a congested parking lot? Parking issues not only affect our emotions but also have an impact on the environment. According to the analysis by INRIX in 2017, American drivers spend an average of 17 hours a year searching for parking spots, which adds up to an estimated $345 per driver in wasted time, fuel, and emissions.

That’s how we came up with our project—ParkinGPT. The initiation of this project is to park in a Greener, Safer, and Smarter way. Using ParkinGPT, we can fully explore the power of the state-of-the-art large language model, GPT-4, which could leverage smarter ways of interacting with navigation applications. With the assistance of GPT-4, users intuitively use natural language conversations to find optimal parking lot options to save more time and fuel. We also incorporate safety factors by using INRIX APIs when choosing the lot, ensuring the user's personal safety.


## What it does

The logistics of our project are the following: The user initially tells ParkinGPT their destination through the AI chat box. ParkinGPT will extract the location from their dialog, search it in Google Map, return several results, and let the user confirm their destination. After that, ParkinGPT will recommend parking lots around the destination by applying INRIX’s off-street parking API. At the same time, it will also pull out the safety alert information around the user’s destination using INRIX’s incidents API to tell whether a recommended parking lot is safe. With this information, ParkinGPT will run a Random Forest Regressor to compute a rating score for each parking lot, sort all the parking lots based on the ratings, choose the top 5 options and provide them to the users. If the user is unsatisfied with the result, they could give more instructions to ParkinGPT through the chat box to regenerate results following the same procedure above.


## How we built it

Our project mainly has 3 parts: machine learning, back-end, and front-end. We will go through them one by one.

**Front-End**

We used React.js in Next.js to implement the entire front-end framework and combine it with Tailwind and Material UI to customize its appearance. Since we focus on finding locations and parking lots, we embedded Google Map into our website and made it the primary tool to visualize all the data processed from the back-end. We provided a chatbot that incorporates GPT-4 as the tool to parse a conversation with the users so that we can enhance our user experience and make the website more interactive. 

**Back-End**

We embedded several APIs from Google Map (Places API), Open AI (...), and INRIX(lots API and incident API) and implemented our own API based on Flask (with our machine learning model). The back-end works like this: it receives data as input queries, and after several data processing, data is then used to send a request to API for information we need. Our back-end structure includes mainly 3 parts: one folder for the definition of parameters being used to send requests, one folder for validation check of input query and parameters, and one folder for API connection, requests sending, data post-processiang, and backward data responding. Through these solid steps, we avoid invalid inputs and fatal errors and clearly organize the transfer of data between the front-end and back-end.

**Machine Learning**

We randomly chose 39 locations located in downtown San Francisco. In each of them, we used off-street INRIX API to search for parking lots within 1000 meters of those locations. In total, we found 3517 different parking lots. We then extracted 6 features out of the JSON output for each parking lot. They include the percentage of occupied inside the facility (in %), the likelihood of finding a parking spot within the facility (in %), the calculated amount of available parking spots, the distance of the parking lot to the user’s destination (in meters), the level of pricing for parking (cannot find docs on the website, the higher the number, the higher the cost, -1 if null), the average rating score of that parking lot (in a scale from 1 to 5, -1 if null). 

After that, among those parking lots’ locations, we randomly chose 31 locations to run incidents INRIX API to search for safety alerts within 100 to 500 meters of those parking lots (since the running time is unstable, we had to adjust the radius). We got 897 different incidents in total. Then, we extracted 4 more features for each parking lot. For each parking lot, consider all incidents happening within 500 meters. We added the severity of each of the four types of incidents and made them into 4 different features to indicate how dangerous a given parking lot is. We then did prompt engineering in detail to teach GPT-4 to learn about our dataset and label our dataset with a consistent methodology. We asked it to go through each parking lot, use feature data, and rate the parking lot on a scale of 16 to 20 with a standard deviation over 1, where we then manually subtracted 15 out of every rating to rectify the scale to 1 to 5 (we did this because when directly telling GPT-4 to rate in a scale of 1 to 5, it will never meet our need). Next, we standardized the feature data using the Z-score approach to promote our future training. We then trained 4 different regression models (Linear Regression, Random Forest Regressor, Gradient Boosting Regressor, and Decision Tree Regressor) with K-Folds of 10. After comparing all models, we found that Random Forest Regressor showed the lowest MSE of 0.102 (or RMSE of 0.319). Finally, in order to connect our model to the server, we built a model prediction API using the Python package Flask.


## Challenges we ran into

**Front-End**

The first challenge in the front-end part was embedding Google Map into our website. There were some problems with obtaining Map Javascript API keys and implementing the Map in React, but we managed to overcome this issue with some searching on the internet. The second challenge was to design and implement the user interface. We were using Material UI, which provides us with lots of tools for great styling. However, we couldn’t figure out how to overwrite some minor changes we didn’t want from the default styling in Material UI. We managed to solve it at last using the important keyword even though the styling was unfavorable. Lastly, we need to connect our chatbox to GPT-4 and use a data structure to store all the past conversations, as we need to print it out to the user. 

**Back-End**

For the back-end part, we faced challenges when we tried to develop our own API. We had little to no experience in developing in Flask. Also, the interaction between our back-end and our own API took us hours to figure out. When using existing APIs provided by tech companies, we are provided detailed documentation about what input and output are expected. To develop and use our own API, we had to define input and output ourselves. We had to balance the difficulties of providing satisfying input to API while suffering from complicated data post-processing and doing as little as possible data processing at the back end while handing over all tasks to API. Unlike using existing API, we had the right to modify and update our own API. Decision-making on such problems can always be torturing, especially when we have no experience in judging what benefited us most.

**Machine Learning**

We mainly met three challenges for the machine learning part of our project. During our data acquisition process, we came across an issue when using the off-street parking INRIX API: the running time of one single search was unexpectedly long. The solution to that is to use three laptops at the same time and decrease the radius of search. In need of over 3500 ground-truth labels for training usage, it was also challenging for us to label them all manually due to the incredibly large workload. To solve that, we made a smart use of the GPT-4 to help us label them all. However, the prompt engineering process was also torturous for us, just like training a toddler. Plus, when trying to post our trained machine learning model to the server, we also struggled to learn how to use Flask, but fortunately, we made it through all the difficulties.


## Accomplishments that we're proud of

1. A massive back-end API proxy that makes connection to the GPT-4 API, AWS API, Google Map API, and INRIX API to enable all the queries the user would like to request. 
2. Created a training dataset by combining the lots API and incidents API of INRIX and trained it to achieve a parking lot recommendation model.
3. Successfully loaded the machine learning model to javascript, making the application smarter.
4. Designed and developed a sophisticated, modern-styled navigation website, incorporating cutting-edge user interface principles and seamless user experience.
5. Made and uploaded a structured collection of codes and datasets on GitHub for us to maintain and keep record.


## What we learned

1. Using, calling, and understanding a variety of APIs, including INRIX API, GPT-4 API, Google Map API, Flask API.
2. Designing, collecting, and processing raw datasets in INRIX API into a structured and formalized dataset.
3. Preparing, training, validating, and applying numerous machine learning models using K-Folds techniques.
4. Designing, creating, and implementing an innovative UI design for the infrastructure of ParkinGPT and better user experiences.
5. Designing, modifying, and building a massive backend structure that utilizes a lot of RESTful APIs to pursue our goals.


## What's next for ParkinGPT

1. Expand the size of diversity of our dataset to increase the competency and robustness of our machine learning models and explore more possibilities of model choices.
2. Enable more utilities including adding the function of giving advice for multiple destinations.
3. Gather more on-streets parking data so that we could use that to provide on-street parking advice (we don’t have data that are detailed enough).
4. Add a transaction function to help user easily and conveniently do the parking payment directly on the application.
5. Record the user’s parking lot preference and use a recommendation algorithm to use the historical parking pattern to make further suggestions.
6. Integrate a speech recognition function to enable users to interact with ParkinGPT in real audio conversation.
7. Use a database to store users’ chat history so that ParkinGPT could provide more personalized services to improve user experiences.
