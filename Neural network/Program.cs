using System;
using System.IO;
using System.Linq;
using System.Collections.Generic;

namespace Neural_network
{
    class Program
    {
        static void Main(string[] args)
        {
            neuron[] inputLayer = new neuron[3]; // these values can be changed to be the parameters for any neural network
            neuron[] hiddenLayer = new neuron[10];
            neuron[] outputLayer = new neuron[9];
            bool direction; // true for forwards, false for backwards
            float[] outputs = new float[outputLayer.Length];
            float bestCost = 100000000.0f;
            int countWithoutImprovment = 0;
            int threshold = 10;

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

            string[] trainingData = File.ReadAllLines("training.txt");
            string[] firstLine = trainingData[0].Split(' ');
            for (int i = 0; i < inputLayer.Length; i++)
            {
                inputLayer[i].inputs[0] = float.Parse(firstLine[i]);
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
            float[]actualOutput = new float[outputLayer.Length];
            for (int i = 0; i < outputLayer.Length; i++)
            {
                actualOutput[i] = outputLayer[i].output;
            }
            float[]expectedOutput = new float[outputLayer.Length];
            expectedOutput[getExpectedOutput(firstLine[3])] = 1.0f;
            float[] differences = new float[expectedOutput.Length];
            for (int i = 0; i < expectedOutput.Length; i++)
            {
                differences[i] = actualOutput[i] - expectedOutput[i];
            }
            float firstCost = 0.0f;
            for (int i = 0; i < differences.Length; i++)
            {
                firstCost += differences[i];
            }

            for (int i = 0; i < inputLayer.Length; i++)
            {
                inputLayer[i].step(true);
            }
            for (int i = 0; i < hiddenLayer.Length; i++)
            {
                hiddenLayer[i].step(true);
            }
            for (int i = 0; i < outputLayer.Length; i++)
            {
                outputLayer[i].step(true);
            }
            
            string[] secondLine = trainingData[1].Split(' ');
            for (int i = 0; i < inputLayer.Length; i++)
            {
                inputLayer[i].inputs[0] = float.Parse(secondLine[i]);
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
            expectedOutput[getExpectedOutput(secondLine[3])] = 1.0f;
            differences = new float[expectedOutput.Length];
            for (int i = 0; i < expectedOutput.Length; i++)
            {
                differences[i] = actualOutput[i] - expectedOutput[i];
            }
            float secondCost = 0.0f;
            for (int i = 0; i < differences.Length; i++)
            {
                secondCost += differences[i];
            }
            
            if (Math.Abs(secondCost) > Math.Abs(firstCost))
            {
                direction = true;
            }
            else
            {
                direction = false;
            }
            for (int i = 0; i < inputLayer.Length; i++)
            {
                inputLayer[i].step(direction);
            }
            for (int i = 0; i < hiddenLayer.Length; i++)
            {
                hiddenLayer[i].step(direction);
            }
            for (int i = 0; i < outputLayer.Length; i++)
            {
                outputLayer[i].step(direction);
            }

            for (int a = 2; a < trainingData.Length-1; a++)
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
                expectedOutput[getExpectedOutput(firstLine[3])] = 1.0f;
                differences = new float[expectedOutput.Length];
                for (int i = 0; i < expectedOutput.Length; i++)
                {
                    differences[i] = actualOutput[i] - expectedOutput[i];
                }
                float cost = 0.0f;
                for (int i = 0; i < differences.Length; i++)
                {
                    cost += differences[i];
                }

                for (int i = 0; i < inputLayer.Length; i++)
                {
                    inputLayer[i].step(direction);
                }
                for (int i = 0; i < hiddenLayer.Length; i++)
                {
                    hiddenLayer[i].step(direction);
                }
                for (int i = 0; i < outputLayer.Length; i++)
                {
                    outputLayer[i].step(direction);
                }
                Console.WriteLine("Current cost: "+cost);
                if (cost < bestCost)
                {
                    bestCost = cost;
                    for (int b = 0; b < inputLayer.Length; b++)
                    {
                        inputLayer[b].bestWeights = inputLayer[b].weights;
                    }
                    for (int b = 0; b < hiddenLayer.Length; b++)
                    {
                        hiddenLayer[b].bestWeights = hiddenLayer[b].weights;
                    }
                    for (int b = 0; b < outputLayer.Length; b++)
                    {
                        outputLayer[b].bestWeights = outputLayer[b].weights;
                    }
                }
                else 
                {
                    countWithoutImprovment++;
                    if (countWithoutImprovment > threshold)
                    {
                        for (int b = 0; b < inputLayer.Length; b++)
                        {
                            inputLayer[b].weights = inputLayer[b].bestWeights;
                        }
                        for (int b = 0; b < hiddenLayer.Length; b++)
                        {
                            hiddenLayer[b].weights = hiddenLayer[b].bestWeights;
                        }
                        for (int b = 0; b < outputLayer.Length; b++)
                        {
                            outputLayer[b].weights = outputLayer[b].bestWeights;
                        }
                        break;
                    }
                }
            }

            string[]testData = File.ReadAllLines("training.txt");
            for (int a = 0; a < testData.Length; a++)
            {
                string[] currLine = testData[a].Split(' ');
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
                Console.WriteLine("Colour guessed: "+getColourName(actualOutput) + " | Actual colour: " + getColourName(expectedOutput));
            }
        }
        public static int getExpectedOutput(string input)
        {
            string[]colours = {"red","orange","yellow","green","blue","purple","white","grey","black"};
            return Array.IndexOf(colours, input);
        }
        public static string getColourName(float[]input)
        {
            string[]colours = {"red","orange","yellow","green","blue","purple","white","grey","black"};
            return colours[input.ToList().IndexOf(input.Max())];
        }
    }
    class neuron
    {
        Random randnum = new Random();
        public float[] inputs;
        public float[] weights;
        float[] stepSizes;
        public float[] bestWeights;
        public float output;
        float bias; // when the bias is added to the summed weighted inputs, the network seems to perform worse

        public void initialize(int numOfInputs)
        {
            inputs = new float[numOfInputs];
            weights = new float[numOfInputs];
            stepSizes = new float[numOfInputs];
            bestWeights = weights;
            bias = (float)randnum.NextDouble() + randnum.Next(10);
            for (int i = 0; i < weights.Length; i++)
            {
                weights[i] = (float)randnum.NextDouble();
            }
            for (int i = 0; i < stepSizes.Length; i++)
            {
                stepSizes[i] = (float)randnum.NextDouble();
                if (randnum.Next(0,2) == 0)
                {
                    stepSizes[i] *= -1;
                }
            }
        }
        public void calculate()
        {
            float total = 0;
            for (int i = 0; i < inputs.Length; i++)
            {
                total += inputs[i] * weights[i];
            }
            //total += bias;
            output = sigmoid(total);
        }
        public void step(bool direction)
        {
            if (direction == true)
            {
                for (int i = 0; i < stepSizes.Length; i++)
                {
                    weights[i] += stepSizes[i];
                }
            }
            else
            {
                for (int i = 0; i < stepSizes.Length; i++)
                {
                    weights[i] -= stepSizes[i];
                }
            }
        }
        public float sigmoid (float input)
        {
            return (float)(1/(1+Math.Exp((double)-input)));
        }
    }
}
