# Functional Oblivious Transfer (FOT)
## Description
This library provides code for: 

  * FOT3: Unconditionally secure functional t-out-of-n OT, where the function is [Mean](https://github.com/anonymous2012000/FOT/blob/main/FOT3--mean.cpp)
  * FOT3: Unconditionally secure functional t-out-of-n OT, where the function is [Mode](https://github.com/anonymous2012000/FOT/blob/main/FOT3--mode.cpp)
  * FOT2: Fully homomrphic-based t-out-of-n OT, where the function is [Mean](https://github.com/anonymous2012000/FOT/blob/main/FOT2--mean.cpp)




## Dependencies

* [GMP](https://gmplib.org/)

## Using the library

1. Clone the above library.
2. Install the library and unzip the file.
3. To run Mode, run the following command lines in order:

 ```
    cd Directory/FOT-main

    g++ -std=c++11 -lgmpxx -lgmp main--mode.cpp -o main

    ./main
```
4. Now, too run Mean, run the following command lines in order (we assume you've already in the related directory):
   
```
    g++ -std=c++11 -lgmpxx -lgmp main--mean.cpp -o main
   
    ./main
```

   
       

      
