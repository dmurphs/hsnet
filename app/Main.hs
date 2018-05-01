{-# LANGUAGE OverloadedStrings #-}

module Main where

import Data.Csv
import Numeric.LinearAlgebra
import qualified Data.ByteString.Lazy as BL
import qualified Data.Vector as V

import Lib

-- data Iris = Iris
--   { sepalLength :: !Double,
--     sepalWidth  :: !Double,
--     petalLength :: !Double,
--     petalWidth  :: !Double,
--     className   :: !String
--   }
-- 
-- encodeClassName :: Iris -> Maybe [Double]
-- encodeClassName i = case className i of
--   "Iris-setosa"     -> Just [1,0,0]
--   "Iris-versicolor" -> Just [0,1,0]
--   "Iris-virginica"  -> Just [0,0,1]
--   _                 -> Nothing
-- 
-- instance FromNamedRecord Iris where
--   parseNamedRecord r = Iris <$> r .: "sepalLength"
--     <*> r .: "sepalWidth"
--     <*> r .: "petalLength"
--     <*> r .: "petalWidth"
--     <*> r .: "className"

main = print "hello"
