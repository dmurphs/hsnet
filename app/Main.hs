{-# LANGUAGE OverloadedStrings #-}

module Main where

import Data.Csv
import Numeric.LinearAlgebra
import qualified Data.ByteString.Lazy as BL
import qualified Data.Vector as V

import Lib

-- Use Iris dataset to test functionality

data Iris = Iris
  { sepalLength :: !Double,
    sepalWidth  :: !Double,
    petalLength :: !Double,
    petalWidth  :: !Double,
    className   :: !String
  }

encodeClassName :: Iris -> [Double]
encodeClassName i = case className i of
  "Iris-setosa"     -> [1,0,0]
  "Iris-versicolor" -> [0,1,0]
  "Iris-virginica"  -> [0,0,1]

instance FromNamedRecord Iris where
  parseNamedRecord r = Iris <$> r .: "sepalLength"
    <*> r .: "sepalWidth"
    <*> r .: "petalLength"
    <*> r .: "petalWidth"
    <*> r .: "className"

irisToInput :: Iris -> [Double]
irisToInput iris = map ($ iris) [sepalLength,sepalWidth,petalLength,petalWidth]

-- Hyperparameters
regularizationTerm=0.01
learningRate=0.02
layerSizes=[4,5,3]

-- Define Model
initializerScale=0.0005
initialBiases = biasesWithInitializer initializerScale layerSizes
initialWeights = weightsWithInitializer initializerScale layerSizes
initialParameters = zip initialWeights initialBiases
activationFunctions = [sigmoid,sigmoid]
layers = zipWith getLayer initialParameters activationFunctions

-- TODO: Write function to get validation set performance

main = do
  irisTrainingData <- BL.readFile "data/training.csv"
  case decodeByName irisTrainingData of
    Left err -> putStrLn err
    Right (_,vTraining) -> do
      let
        irisTrainingList = map irisToInput $ V.toList vTraining
        targetOutputList = map encodeClassName $ V.toList vTraining
        irisTrainingInputMatrix = fromLists irisTrainingList
        irisTrainingTargetOutputMatrix = fromLists targetOutputList
        updatedParameters = runEpochs 3000 learningRate initializerScale irisTrainingInputMatrix irisTrainingTargetOutputMatrix layers
        updatedLayers = zipWith getLayer updatedParameters activationFunctions
      irisTestData <- BL.readFile "data/test.csv"
      case decodeByName irisTestData of
        Left err -> putStrLn err
        Right (_,vTest) -> do
          let
            irisTestList = map irisToInput $ V.toList vTest
            targetOutputList = map encodeClassName $ V.toList vTest
            irisTestInputMatrix = fromLists irisTestList
            irisTestTargetOutputMatrix = fromLists targetOutputList
            testActivations = forwardPropogate irisTestInputMatrix updatedLayers
            outputs = (postActivation . last) testActivations :: Matrix Double
            targetOutputRows = toRows irisTestTargetOutputMatrix
            outputRows = toRows outputs
            targetArgmaxList = map maxIndex targetOutputRows
            outputArgmaxList = map maxIndex outputRows
            zippedArgmaxLists = zip targetArgmaxList outputArgmaxList
            total = length zippedArgmaxLists
            correct = length [(a,b) | (a,b) <- zippedArgmaxLists,a == b]
          print total
          print correct
          print outputs
          print irisTestTargetOutputMatrix

