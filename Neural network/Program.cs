using System;
using System.IO;
using System.Linq;
using System.Collections.Generic;

namespace Neural_network
{
    class Program
    {
        static string[] colours = {"red","orange","yellow","green","blue","purple","white","black"};
        static void Main(string[] args)
        {
            neuron[] inputLayer = new neuron[3]; // these values can be changed to be the parameters for any neural network
            neuron[] hiddenLayer = new neuron[20];
            neuron[] outputLayer = new neuron[colours.Length];
            float[]actualOutput;
            float[]expectedOutput;
            const string trainingPath = "training.txt";
            string[] trainingData = File.ReadAllLines(trainingPath);
            // initiaization
            Console.Clear();
            for (int i = 0; i < inputLayer.Length;i++)
            {
                inputLayer[i] = new neuron();
            }
            for (int i = 0; i < hiddenLayer.Length;i++)
            {
                hiddenLayer[i] = new neuron();
            }
            for (int i = 0; i < outputLayer.Length;i++)
            {
                outputLayer[i] = new neuron();
            }
            
            for (int i = 0; i < inputLayer.Length; i++)
            {
                inputLayer[i].initialize(1);
            }
            for (int i = 0; i < hiddenLayer.Length; i++)
            {
                hiddenLayer[i].initialize(inputLayer.Length);
            }
            for (int i = 0; i < outputLayer.Length; i++)
            {
                outputLayer[i].initialize(hiddenLayer.Length);
            }
            // training
            Console.WriteLine("Training weights");
            for (int x = 0; x < 10000; x++)
            {
                for (int a = 0; a < trainingData.Length; a++)
                {
                    string[] currLine = trainingData[a].Split(' ');
                    for (int i = 0; i < inputLayer.Length; i++)
                    {
                        inputLayer[i].inputs[0] = float.Parse(currLine[i]);
                    }
                    for (int i = 0; i < inputLayer.Length; i++)
                    {
                        inputLayer[i].calculate();
                    }
                    for (int i = 0; i < hiddenLayer.Length; i++)
                    {
                        for (int j = 0; j < hiddenLayer[i].inputs.Length; j++)
                        {
                            hiddenLayer[i].inputs[j] = inputLayer[j].output;
                        }
                    }
                    for (int i = 0; i < hiddenLayer.Length; i++)
                    {
                        hiddenLayer[i].calculate();
                    }
                    for (int i = 0; i < outputLayer.Length; i++)
                    {
                        for (int j = 0; j < outputLayer[i].inputs.Length; j++)
                        {
                            outputLayer[i].inputs[j] = hiddenLayer[j].output;
                        }
                    }
                    for (int i = 0; i < outputLayer.Length; i++)
                    {
                        outputLayer[i].calculate();
                    }
                    actualOutput = new float[outputLayer.Length];
                    for (int i = 0; i < outputLayer.Length; i++)
                    {
                        actualOutput[i] = outputLayer[i].output;
                    }
                    expectedOutput = new float[outputLayer.Length];
                    expectedOutput[getExpectedOutput(currLine[3])] = 1.0f;
                    float[] inputLayerOutputs = new float[inputLayer.Length];
                    for (int b = 0; b < inputLayer.Length; b++)
                    {
                        inputLayerOutputs[b] = inputLayer[b].output;
                    }
                    for (int b = 0; b < outputLayer.Length; b++)
                    {
                        outputLayer[b].adjustOutputWeights(expectedOutput[b],actualOutput[b]);
                    }
                    for (int b = 0; b < hiddenLayer.Length; b++)
                    {
                        hiddenLayer[b].adjustHiddenWeights(expectedOutput,actualOutput);
                    }
                    //Console.WriteLine(getColourName(expectedOutput) + " "+getColourName(actualOutput));
                }
            }

            Console.WriteLine("Testing");
            // testing
            while(true)
            {
                Console.Write("Enter an RGB value: ");
                string [] userInputStr = Console.ReadLine().Split(' ');
                for (int i = 0; i < userInputStr.Length; i++)
                {
                    inputLayer[i].inputs[0] = float.Parse(userInputStr[i])/255;
                }
                for (int i = 0; i < inputLayer.Length; i++)
                { 
                    inputLayer[i].calculate();
                }
                for (int i = 0; i < hiddenLayer.Length; i++)
                {
                    for (int j = 0; j < hiddenLayer[i].inputs.Length; j++)
                    {
                        hiddenLayer[i].inputs[j] = inputLayer[j].output;
                    }
                }
                for (int i = 0; i < hiddenLayer.Length; i++)
                {
                    hiddenLayer[i].calculate();
                }
                for (int i = 0; i < outputLayer.Length; i++)
                {
                    for (int j = 0; j < outputLayer[i].inputs.Length; j++)
                    {
                        outputLayer[i].inputs[j] = hiddenLayer[j].output;
                    }
                }
                for (int i = 0; i < outputLayer.Length; i++)
                {
                    outputLayer[i].calculate();
                }
                actualOutput = new float[outputLayer.Length];
                for (int i = 0; i < outputLayer.Length; i++)
                {
                    actualOutput[i] = outputLayer[i].output;
                }
                Console.WriteLine("Colour guessed: "+getColourName(actualOutput));
            }
        }
        public static int getExpectedOutput(string input)
        {
            return Array.IndexOf(colours, input);
        }
        public static string getColourName(float[] input)
        {
            return colours[Array.IndexOf(input, input.Max())];
        }
    }
    class neuron
    {
        Random randnum = new Random();
        public float[] inputs;
        public float[] weights;
        public float output;
        public float bias;
        static float[] signModifier = {-1.0f,1.0f};
        const float learningRate = 0.1f;

        public void initialize(int numOfInputs)
        {
            inputs = new float[numOfInputs];
            weights = new float[numOfInputs];

            bias = 10.0f*(float)randnum.NextDouble()*signModifier[randnum.Next(0,2)];
            for (int i = 0; i < weights.Length; i++)
            {
                weights[i] = 1.0f*(float)randnum.NextDouble()*signModifier[randnum.Next(0,2)];
            }
        }
        public void calculate()
        {
            float total = 0;
            for (int i = 0; i < inputs.Length; i++)
            {
                total += inputs[i] * weights[i];
            }
            total += bias;
            output = sigmoid(total);
        }
        public void adjustOutputWeights(float actualOutput,float calculatedOutput)
        {
            float factor = (actualOutput-calculatedOutput)*calculatedOutput*(1-calculatedOutput);
            for (int i = 0; i < weights.Length; i++)
            {
                weights[i] += factor*inputs[i];
            }
            bias += factor;
        }
        public void adjustHiddenWeights (float[]actualOutput,float[] calculatedOutput)
        {
            float resultOfSum = 0.0f;
            for (int i = 0; i < actualOutput.Length; i++)
            {
                resultOfSum += -(actualOutput[i]-calculatedOutput[i])*calculatedOutput[i]*(1-calculatedOutput[i]);
            }
            for (int i = 0; i < weights.Length; i++)
            {
                weights[i] -= learningRate*resultOfSum*output*(1-output)*inputs[i];
            }
        }

        public float sigmoid (float input)
        {
            return (float)(1/(1+Math.Exp((double)-input)));
        }
    }
}
