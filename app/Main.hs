{-# LANGUAGE OverloadedStrings #-}

module Main where

import Data.Csv
import Numeric.LinearAlgebra
import qualified Data.ByteString.Lazy as BL
import qualified Data.Vector as V

import Lib

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

regularizationTerm = 0.01
learningRate = 0.02
weightInitializer = 0.0005
layerSizes = [4,5,3]

main = do
  irisTrainingData <- BL.readFile "data/training.csv"
  case decodeByName irisTrainingData of
    Left err -> putStrLn err
    Right (_,vTraining) -> do
      let
        irisTrainingList = concatMap irisToInput $ V.toList vTraining
        expectedOutputList = concatMap encodeClassName $ V.toList vTraining
        irisTrainingInputMatrix = (125><4) irisTrainingList
        irisTrainingExpectedOutputMatrix = (125><3) expectedOutputList
        initialWeights = weightsWithInitializer weightInitializer layerSizes
        -- TODO: add function to initialize biases
        initialBiases = [Numeric.LinearAlgebra.fromList $ replicate 5 0.0005, Numeric.LinearAlgebra.fromList $ replicate 3 0.0005] :: [Vector Double]
        initialParameters = zip initialWeights initialBiases
        activationFunctions = [sigmoid,sigmoid]
        layers = zipWith getLayer initialParameters activationFunctions
        updatedParameters = runEpochs 5000 regularizationTerm learningRate irisTrainingInputMatrix irisTrainingExpectedOutputMatrix layers
        updatedLayers = zipWith getLayer updatedParameters activationFunctions
      irisTestData <- BL.readFile "data/test.csv"
      case decodeByName irisTestData of
        Left err -> putStrLn err
        Right (_,vTest) -> do
          let
            irisTestList = concatMap irisToInput $ V.toList vTest
            expectedOutputList = concatMap encodeClassName $ V.toList vTest
            irisTestInputMatrix = (25><4) irisTestList
            irisTestExpectedOutputMatrix = (25><3) expectedOutputList
            testActivations = forwardPropogate irisTestInputMatrix updatedLayers
            outputs = (postActivation . last) testActivations :: Matrix Double
          print outputs
          print irisTestExpectedOutputMatrix

