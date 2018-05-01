module Lib
    ( forwardPropogate,
      weightsWithInitializer,
      randomWeights,
      sigmoid,
      runBatch,
      tupleToLayer,
      ActivationFunction (..),
      Activation (..)
    ) where

import Numeric.LinearAlgebra

-- |The Layer type represents a layer in an artificial neural network
data Layer = Layer { weights            :: Matrix Double
                   , biases             :: Vector Double
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

-- |Turns a tuple of form (weights,biases,activationFunction) into a Layer
tupleToLayer :: (Matrix Double, Vector Double, ActivationFunction) -> Layer
tupleToLayer (layerWeights,layerBiases,layerActivationFunction) =
  Layer { weights=layerWeights, biases=layerBiases, activationFunction = layerActivationFunction }

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
    layerWeights = weights layer
    layerBiases = biases layer
    layerActivationFunction = (f . activationFunction) layer

-- |Get random weights for a given network structure
randomWeights :: [Int] -> Maybe (IO [Matrix Double])
randomWeights []  = Nothing
randomWeights [x] = Nothing
randomWeights xs  = Just $ traverse (uncurry randn) $ steps xs

-- |Get weights all with initializer value for given network structure
weightsWithInitializer :: Double -> [Int] -> [Matrix Double]
weightsWithInitializer initializer layerSizes = map (\(from,to) -> (from><to) $ repeat initializer) $ steps layerSizes

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
runBatch :: Matrix Double -> Matrix Double -> [Layer] -> [(Matrix Double, Vector Double)]
runBatch inputs expected initialParameters = runBatch' (tail reversedActivations) reversedInititalWeights (tail activationDerivatives) outputDelta [(outputWeightGradients, outputBiasGradients)]
  where
    runBatch' (currentActivations:previousActivations:remaining) (nextWeights:currentWeights) (currentActivationDerivative:remainingActivationDerivatives) nextDelta gradients = runBatch' (previousActivations:remaining) currentWeights remainingActivationDerivatives currentDelta (gradient:gradients)
      where
        gradient = (getWeightGradients batchSize currentDelta previousActivations, getBiasGradient currentDelta)
        currentDelta = getDelta currentActivations nextDelta nextWeights currentActivationDerivative
    runBatch' [input] _ _ _ gradients = gradients
    outputBiasGradients = getBiasGradient outputDelta
    outputWeightGradients = getWeightGradients batchSize outputDelta outputPreviousActivations
    outputDelta = getOutputDelta expected output outputActivationDerivative 
    output = head reversedActivations
    outputPreviousActivations = reversedActivations !! 1
    outputActivationDerivative = last activationDerivatives
    reversedActivations = reverse activations
    activations = forwardPropogate inputs initialParameters
    reversedInititalWeights = reverse initialWeights
    initialWeights = map weights initialParameters
    initialBiases = map biases initialParameters
    activationDerivatives = map (f' . activationFunction) initialParameters
    batchSize = rows inputs

getDelta :: Activation -> Matrix Double -> Matrix Double -> (Double -> Double) -> Matrix Double
getDelta activation nextDelta nextWeights activationDerivative = (nextDelta <> tr nextWeights) * cmap activationDerivative (preActivation activation)

getBiasGradient :: Matrix Double -> Vector Double
getBiasGradient deltaMatrix = fromList $ map (\column -> sum column / fromIntegral (length column)) columns
  where
    columns = map toList $ toColumns deltaMatrix

getOutputDelta :: Matrix Double -> Activation -> (Double -> Double) -> Matrix Double
getOutputDelta expected output activationDerivative = (expected - postActivation output) * cmap activationDerivative (preActivation output)

getWeightGradients :: Int -> Matrix Double -> Activation -> Matrix Double
getWeightGradients batchSize delta previousActivations = (tr previousPostActivation <> delta) / fromIntegral batchSize
  where
    previousPostActivation = postActivation previousActivations


