{-# LANGUAGE BangPatterns      #-}
{-# LANGUAGE CPP               #-}
{-# LANGUAGE NoImplicitPrelude #-}

#if __GLASGOW_HASKELL__ >= 702
{-# LANGUAGE Trustworthy       #-}
#endif

{-# OPTIONS_GHC -Wall #-}

-- |
-- Copyright: © Herbert Valerio Riedel 2017-2018
-- SPDX-License-Identifier: BSD-3-Clause
--
-- This module is implicitly imported in all modules unless @{-\# LANGUAGE NoImplicitPrelude \#-}@ is in effect.
--
module Prelude (
    -- ** 'Char'
    Char, String, IsString(fromString)

    -- ** 'Bool'
  , Bool(False, True)
  , (&&), (||), not, otherwise, bool

    -- ** 'Maybe'
  , Maybe(Nothing, Just)
  , maybe, fromMaybe, isJust, isNothing, mapMaybe, catMaybes, listToMaybe, maybeToList

    -- ** 'Either'
  , Either(Left, Right)
  , either, fromLeft, fromRight, isLeft, isRight, lefts, rights, partitionEithers

    -- ** 'Eq' and 'Ord'
  , Ordering(LT, EQ, GT)
  , Eq((==), (/=))
  , Ord(compare, (<), (<=), (>=), (>), max, min)
  , comparing

    -- ** 'Enum'
  , Enum(fromEnum, enumFrom, enumFromThen, enumFromTo, enumFromThenTo)

    -- ** 'Bounded'
  , Bounded(minBound, maxBound)

    -- ** 2-tuples
  , fst, snd, curry, uncurry

    -- ** 'Num'eric types, classes, and helpers
  , Integer
  , Int, Int8, Int16, Int32, Int64
  , Word, Word8, Word16, Word32, Word64
  , Float, Double, Rational

  , Num((+), (-), (*), negate, abs, signum, fromInteger)
  , Real(toRational)
  , Integral(quot, rem, div, mod, quotRem, divMod, toInteger)
  , Fractional((/), recip, fromRational)
  , Floating(pi, exp, log, sqrt, (**), logBase, sin, cos, tan, asin, acos, atan, sinh, cosh, tanh, asinh, acosh, atanh)
  , RealFrac(properFraction, truncate, round, ceiling, floor)
  , RealFloat(floatRadix, floatDigits, floatRange, decodeFloat, encodeFloat, exponent, significand, scaleFloat, isNaN, isInfinite, isDenormalized, isIEEE, isNegativeZero, atan2)

  , subtract, even, odd, gcd, lcm, (^), (^^)
  , fromIntegral, realToFrac

    -- ** 'Semigroup' and 'Monoid'
  , Semigroup((<>)), sconcat
  , Monoid(mempty, mappend, mconcat)

    -- ** 'Functor', 'Applicative', 'Monad', and 'Alternative'
  , Fun.Functor(fmap, (<$)), (<$>)
  , Applicative(pure, (<*>), (*>), (<*))
  , Monad((>>=), (>>), return)
  , Alternative(empty, (<|>), some, many)
  , MonadFail
  , MonadPlus(mzero,mplus)
  , guard, ($>), Trav.forM, F.forM_, F.mapM_, F.sequence_, (=<<), when, unless, void, F.msum, replicateM, replicateM_, forever, (>=>), (<=<), foldM, foldM_, join
  , liftM, liftM2, liftM3, liftM4, liftM5
  , liftA, liftA2, liftA3, liftA4, liftA5

    -- ** 'Foldable' and 'Traversable'
  , F.Foldable(foldMap, fold, foldr, foldl), F.foldr', F.foldl', F.elem, product, sum, F.toList, F.find

  , Traversable(traverse, sequenceA, mapM, sequence)
  , F.and, F.or, F.any, F.all, F.concat, F.concatMap, F.asum, F.sequenceA_, F.traverse_, F.notElem

  , null, length

    -- ** 'Data', 'Typeable', and 'Generic'

  , Data
  , Typeable
#if MIN_VERSION_base(4,5,0)
  , Generic
#endif

    -- ** List operations
  , Prelude.head
  , Prelude.last
  , Prelude.init
  , Prelude.tail
  , List.inits
  , List.tails
  , uncons

  , List.drop
  , List.dropWhile
  , List.take
  , List.takeWhile

  , List.break
  , List.span
  , List.splitAt

  , List.filter
  , List.iterate
  , List.lookup
  , List.partition
  , List.repeat
  , List.replicate
  , List.reverse
  , List.groupBy
  , List.intersperse
  , List.intercalate

  , List.scanl
  , List.scanl1
  , List.scanr
  , List.scanr1

    -- *** Zipping
  , List.zip
  , List.zip3
  , List.zip4
  , List.zip5
  , List.zip6
  , List.zip7
  , List.zipWith
  , List.zipWith3
  , List.zipWith4
  , List.zipWith5
  , List.zipWith6
  , List.zipWith7
  , List.unzip
  , List.unzip3
  , List.unzip4
  , List.unzip5
  , List.unzip6
  , List.unzip7

    -- ** 'NonEmpty' lists
  , NonEmpty((:|))

    -- ** 'Read' and 'Show'
  , Show(showsPrec, showList, show)
  , ShowS, shows, showChar, showString, showParen

  , Read(readsPrec, readList)
  , ReadS, reads, readParen, Prelude.read

    -- ** 'IO' and 'MonadIO'
  , IO
  , MonadIO(liftIO)

    -- *** Standard input/output
  , putChar, putStr, putStrLn, print
  , getChar, getLine, getContents

    -- *** 'FilePath'
  , FilePath
  , readFile, writeFile, appendFile


    -- ** Common miscellaneous verbs
  , id, const, (.), flip, ($), ($!), (&), until, asTypeOf, seq, on

    -- *** Intentionally partial functions
  , error, errorWithoutStackTrace, undefined

  ) where

import           Control.Applicative          as App
import           Control.Monad
import           Control.Monad.Fail
import           Control.Monad.IO.Class
import           Data.Bool
import           Data.Data                    (Data)
import           Data.Either
import           Data.Foldable                as F
import           Data.Function
import           Data.Functor                 as Fun
import           Data.Int
import qualified Data.List                    as List
import           Data.List.NonEmpty           (NonEmpty (..))
import           Data.Maybe
import           Data.Ord
import           Data.Semigroup
import           Data.String
import           Data.Traversable             as Trav
import           Data.Tuple
import           Data.Typeable                (Typeable)
import           Data.Word
import           System.IO
import           Text.Read

import           GHC.Base
import           GHC.Enum
import           GHC.Float
#if MIN_VERSION_base(4,5,0)
import           GHC.Generics                 (Generic)
#endif
import           GHC.Num
import           GHC.Real
import           GHC.Show

#if MIN_VERSION_base(4,8,0)
import           Data.List                    (uncons)
#endif

#if !MIN_VERSION_base(4,7,0)
import           GHC.Err                      (undefined)
#endif

#if !MIN_VERSION_base(4,6,0)
import           Text.ParserCombinators.ReadP as P
#endif

-- | Last element of a list.
last :: [a] -> Maybe a
last [] = Nothing
last xs = Just (List.last xs)

-- | First element of a list.
head :: [a] -> Maybe a
head []    = Nothing
head (x:_) = Just x

-- | List with the last element removed.
init :: [a] -> Maybe [a]
init [] = Nothing
init xs = Just (List.init xs)

-- | List with the first element removed.
tail :: [a] -> Maybe [a]
tail []     = Nothing
tail (_:xs) = Just xs

-- | Lift a 4-ary function to actions.
liftA4 :: Applicative f => (a -> b -> c -> d -> e) -> f a -> f b -> f c -> f d -> f e
liftA4 f a b c d = liftA3 f a b c <*> d

-- | Lift a 5-ary function to actions.
liftA5 :: Applicative f => (a -> b -> c -> d -> e -> g) -> f a -> f b -> f c -> f d -> f e -> f g
liftA5 f a b c d e = liftA4 f a b c d <*> e


#if !MIN_VERSION_base(4,10,0)
fromLeft :: a -> Either a b -> a
fromLeft _ (Left a) = a
fromLeft a _        = a

fromRight :: b -> Either a b -> b
fromRight _ (Right b) = b
fromRight b _         = b
#endif

#if !MIN_VERSION_base(4,9,0)
errorWithoutStackTrace :: String -> a
errorWithoutStackTrace = error
#endif

#if !MIN_VERSION_base(4,8,0)
infixl 1 &

(&) :: a -> (a -> b) -> b
x & f = f x

null :: Foldable f => f a -> Bool
null = List.null . F.toList

length :: Foldable f => f a -> Int
length = List.length . F.toList

infixr 0 $!

($!)    :: (a -> b) -> a -> b
f $! x  = let !vx = x in f vx

uncons :: [a] -> Maybe (a, [a])
uncons []     = Nothing
uncons (x:xs) = Just (x, xs)
#endif

#if !MIN_VERSION_base(4,7,0)
isRight, isLeft :: Either a b -> Bool
isRight (Left  _) = False
isRight (Right _) = True
isLeft  (Left  _) = True
isLeft  (Right _) = False

bool :: a -> a -> Bool -> a
bool f _ False = f
bool _ t True  = t

infixl 4 $>
($>) :: Functor f => f a -> b -> f b
x $> y = y <$ x
#endif

-- | Parse using the 'Read' instance. Succeeds if there's a unique valid parse; returns 'Nothing' otherwise.
read :: Read a => String -> Maybe a
#if !MIN_VERSION_base(4,6,0)
read s = case [ x | (x,"") <- readPrec_to_S read' minPrec s ] of
           [x] -> Just x
           _   -> Nothing
 where
  read' = do x <- readPrec
             lift P.skipSpaces
             return x
#else
read = readMaybe
#endif
