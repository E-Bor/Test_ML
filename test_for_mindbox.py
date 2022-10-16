import collections


def distribution_of_buyers1(n_customers: int) -> dict:
    # List generation. Map() is used because it is faster than nested list comprehensions
    list_with_sums = [sum(map(int, str(i))) for i in range(n_customers)]
    # Element count from collections module
    sum_counter = collections.Counter(list_with_sums)
    return dict(sum_counter)


def distribution_of_buyers2(n_customers, n_first_id: int) -> dict:
    # List generation. Map() is used because it is faster than nested list comprehensions
    list_with_sums = [sum(map(int, str(i))) for i in range(n_first_id, n_customers)]
    # Element count from collections module
    sum_counter = collections.Counter(list_with_sums)
    return dict(sum_counter)


distribution_of_buyers1(100000)
distribution_of_buyers2(100000, 15)
