from dataset.generator import ArtificialDataset

def main() -> None:
    dataset = ArtificialDataset(4)
    
    for v in dataset._generator(2, 10):
        print(v)


    # a = dataset.batch(3, drop_remainder=False)

    # for value in a:
    #     print(value['enc_in'].shape)
    #     print(value['dec_in'].shape)
    #     print(value['dec_out'].shape)

    # print(a)
    return 1

if __name__ == "__main__":
    main()
