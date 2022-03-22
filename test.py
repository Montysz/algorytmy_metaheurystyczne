import timeit

cy = timeit.timeit('cyth.test2(5)', setup='import cyth', number = 100)
py = timeit.timeit('cyth.test(5)', setup='import cyth', number = 100)