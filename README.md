# Functional Oblivious Transfer (FOT)
## Description
This library provides code for: 

  * FOT3: Unconditionally secure functional t-out-of-n OT, where the function is [Mean](https://github.com/anonymous2012000/FOT/blob/main/FOT3--mean.cpp)
  * FOT3: Unconditionally secure functional t-out-of-n OT, where the function is [Mode](https://github.com/anonymous2012000/FOT/blob/main/FOT3--mode.cpp)
  * FOT2: Fully homomrphic-based t-out-of-n OT, where the function is [Mean](https://github.com/anonymous2012000/FOT/blob/main/FOT2--mean.cpp)




## Dependencies

* [GMP](https://gmplib.org/)
* [HElib](https://github.com/homenc/HElib) (if FOT2 is used)

## Using the library

1. Clone the above library.
2. Install the library and unzip the file.
3. Run the following command lines to run:
  
* Mode (based on FOT3): 

 ```
    cd Directory/FOT-main

    g++ -std=c++11 -lgmpxx -lgmp FOT3--mode.cpp -o main

    ./main

* Mean (based on FOT3): 

   
```
    g++ -std=c++11 -lgmpxx -lgmp FOT3--mean.cpp -o main
   
    ./main

* Mean (based on FOT2): 
   ```
    cd Directory/FOT-main

    g++ -std=c++17 FOT2--mean.cpp  -lhelib -lntl -lgmp -o main 

    ./main


      
