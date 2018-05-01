import Numeric.LinearAlgebra
import Test.Hspec

import Lib 
    ( forwardPropogate,
      sigmoid,
      weightsWithInitializer,
      runBatch,
      tupleToLayer,
      ActivationFunction (..),
      Activation (..)
    )

main :: IO ()
main = hspec $ do
  describe "Lib.forwardPropogate" $ do
    it "correctly performs forward propogation" $ do
      let
        activationFunction = ActivationFunction { f=flip (/) 100, f'= \_ -> 1/100 }
        input    = (1><3) [1..]
        layers   = map tupleToLayer [((3><4) [1..], fromList [1,1,1,1], activationFunction), ((4><2) [1..], fromList [2,2], activationFunction)]
        expected = [input, (1><4) [0.39,0.45,0.51,0.57], (1><2) [10.28 / 100, 12.2 / 100]] :: [Matrix Double]
        activations = forwardPropogate input layers
      map postActivation activations `shouldBe` expected
  describe "Lib.runBatch" $ do
    it "correctly backpropogates" $ do
      let
        input = (1><1) [1] :: Matrix Double
        expected = (1><1) [1]
        initialWeights = weightsWithInitializer 0.05 [1,1,1]
        initialBiases = map fromList [[0.05], [0.05]] :: [Vector Double]
        zippedInitialParameters = zip3 initialWeights initialBiases [sigmoid,sigmoid]
        initialParameters = map tupleToLayer zippedInitialParameters
        expectedGradients = [((1><1) [ 1.4970312464482186e-3 ], fromList [1.4970312464482186e-3]),((1><1) [ 6.303013286946654e-2 ], fromList [0.12006215555353811])]
        actualGradients = runBatch input expected initialParameters
      actualGradients `shouldBe` expectedGradients
