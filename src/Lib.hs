module Lib
    ( forwardPropogate,
      weightsWithInitializer,
      biasesWithInitializer,
      randomWeights,
      sigmoid,
      runBatch,
      updateParameters,
      getLayer,
      runEpochs,
      ActivationFunction (..),
      Activation (..),
      Parameters,
      Gradients,
      Layer (..)
    ) where

import Numeric.LinearAlgebra

-- |The Layer type represents a layer in an artificial neural network
data Layer = Layer { layerParameters  :: Parameters
                   , activationFunction :: ActivationFunction
                   }

-- |Wraps activation function and derivative
data ActivationFunction = ActivationFunction { f  :: Double -> Double
                                             , f' :: Double -> Double
                                             }
-- |Wraps values from layer activations
data Activation = Activation { -- | Values after matrix multiplication but before activation function is applied
                               preActivation     :: Matrix Double
                               -- | Values after activation function is applied
                             , postActivation    :: Matrix Double
                             }

type Parameters = (Matrix Double, Vector Double)
type Gradients = Parameters

-- |Computes values for each layer
forwardPropogate :: Matrix Double -> [Layer] -> [Activation]
forwardPropogate inputs = scanl activate firstActivation
  where
    firstActivation = Activation {preActivation=(numInputRows><numInputCols) [0..], postActivation=inputs}
    numInputRows = rows inputs
    numInputCols = cols inputs

activate :: Activation -> Layer -> Activation
activate previousActivation layer = Activation {preActivation=layerPreActivation, postActivation=layerPostActivation}
  where
    layerPostActivation = cmap layerActivationFunction layerPreActivation
    layerPreActivation = (previousPostActivation <> layerWeights) `add` fromRows (replicate (rows previousPostActivation) layerBiases)
    previousPostActivation = postActivation previousActivation
    (layerWeights,layerBiases) = layerParameters layer 
    layerActivationFunction = (f . activationFunction) layer

-- |Get random weights for a given network structure
randomWeights :: [Int] -> Maybe (IO [Matrix Double])
randomWeights []  = Nothing
randomWeights [x] = Nothing
randomWeights xs  = Just $ traverse (uncurry randn) $ steps xs

-- |Get weights all with initializer value for given network structure
weightsWithInitializer :: Double -> [Int] -> [Matrix Double]
weightsWithInitializer initializer layerSizes = map (\(from,to) -> (from><to) $ repeat initializer) $ steps layerSizes

biasesWithInitializer :: Double -> [Int] -> [Vector Double]
biasesWithInitializer initializer layerSizes = map ((fromList . ($ initializer)) . replicate) $ tail layerSizes

steps :: [a] -> [(a,a)]
steps [] = []
steps [x] = []
steps (x:y:xs) = (x,y) : steps (y:xs)

-- |Sigmoid activation function
sigmoid :: ActivationFunction
sigmoid = ActivationFunction {f=s,f'=s'}
  where
    s x = 1 / (1 + exp (-x))
    s' x = s x * (1 - s x)

-- |Run a batch for training neural network model
runBatch :: Matrix Double -> Matrix Double -> [Layer] -> [Gradients]
runBatch inputs expected initialParameters = runBatch' (outputPreviousActivations:remainingActivations) (reverse initialWeights) (tail reversedActivationDerivatives) outputDelta [(outputWeightGradients, outputBiasGradients)]
  where
    outputBiasGradients = getBiasGradient outputDelta
    outputWeightGradients = getWeightGradients outputDelta outputPreviousActivations
    outputDelta = getOutputDelta expected output outputActivationDerivative 
    (output:outputPreviousActivations:remainingActivations) = reverse activations
    outputActivationDerivative = head reversedActivationDerivatives
    activations = forwardPropogate inputs initialParameters
    initialWeights = map (fst . layerParameters) initialParameters
    initialBiases = map (snd . layerParameters) initialParameters
    reversedActivationDerivatives = reverse $ map (f' . activationFunction) initialParameters

runBatch' :: [Activation] -> [Matrix Double] -> [Double -> Double] -> Matrix Double -> [Gradients] -> [Gradients]
runBatch' [input] _ _ _ gradients = gradients
runBatch' reversedActivations reversedWeights activationDerivatives nextDelta gradients = runBatch' (previousActivations:remaining) currentWeights remainingActivationDerivatives currentDelta (gradient:gradients)
  where
    gradient = (getWeightGradients currentDelta previousActivations, getBiasGradient currentDelta)
    currentDelta = getDelta currentActivations nextDelta nextWeights currentActivationDerivative
    (currentActivations:previousActivations:remaining) = reversedActivations
    (nextWeights:currentWeights) = reversedWeights
    (currentActivationDerivative:remainingActivationDerivatives) = activationDerivatives

getDelta :: Activation -> Matrix Double -> Matrix Double -> (Double -> Double) -> Matrix Double
getDelta activation nextDelta nextWeights activationDerivative = (nextDelta <> tr nextWeights) * cmap activationDerivative (preActivation activation)

getBiasGradient :: Matrix Double -> Vector Double
getBiasGradient deltaMatrix = fromList $ map (\column -> sum column / fromIntegral (length column)) columns
  where
    columns = map toList $ toColumns deltaMatrix

getOutputDelta :: Matrix Double -> Activation -> (Double -> Double) -> Matrix Double
getOutputDelta expected output activationDerivative = (postActivation output - expected) * cmap activationDerivative (preActivation output)

getWeightGradients ::  Matrix Double -> Activation -> Matrix Double
getWeightGradients delta previousActivations = tr previousPostActivation <> delta
  where
    previousPostActivation = postActivation previousActivations

updateParameters :: Double -> Double -> [Gradients] -> [Parameters] -> [Parameters]
updateParameters learningRate regularizationTerm = zipWith updateParameters
  where
    updateParameters :: Gradients -> Parameters -> Parameters
    updateParameters (weightGradients,biasGradients) (weights,biases) =
      (
        weights - cmap (*learningRate) (applyRegularization regularizationTerm weightGradients),
        biases - cmap (*learningRate) biasGradients
      )

applyRegularization :: Double -> Matrix Double -> Matrix Double
applyRegularization regularizationTerm weightGradients = weightGradients + cmap (*regularizationTerm) weightGradients

runEpochs :: Int -> Double -> Double -> Matrix Double -> Matrix Double -> [Layer] -> [Parameters]
runEpochs numEpochs learningRate regularizationTerm inputs expected = runEpochs' numEpochs
  where
    runEpochs' 0 layers = map layerParameters layers
    runEpochs' remainingEpochs layers = runEpochs' (remainingEpochs - 1) (runEpoch learningRate regularizationTerm layers inputs expected)

runEpoch :: Double -> Double -> [Layer] -> Matrix Double -> Matrix Double -> [Layer]
runEpoch learningRate regularizationTerm layers inputs expected = updatedLayers
  where
    gradientMatrices = runBatch inputs expected layers
    parameters = map layerParameters layers
    functions = map activationFunction layers
    updatedParameters = updateParameters learningRate regularizationTerm gradientMatrices parameters
    updatedLayers = zipWith getLayer updatedParameters functions

getLayer :: Parameters -> ActivationFunction -> Layer
getLayer parameters activationFunction = Layer{layerParameters=parameters, activationFunction=activationFunction}
