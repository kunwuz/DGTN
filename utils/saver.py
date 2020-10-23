def print_txt(base_path, args, results, epochs, top_k, note=None, save_config=True):
    path = base_path + "\Best_result_top-"+str(top_k)+".txt"
    outfile = open(path, 'w')
    if note is not None:
        outfile.write("Note:\n"+note+"\n")
    if save_config:
        outfile.write("Configs:\n")
        for attr, value in sorted(args.__dict__.items()):
            outfile.write("{} = {}\n".format(attr, value))

    outfile.write('\nBest results:\n')
    outfile.write("Mrr@{}:\t{}\tEpoch: {}\n".format(top_k, results[1], epochs[1]))
    outfile.write("Recall@{}:\t{}\tEpoch: {}\n".format(top_k, results[0], epochs[0]))
    outfile.close()
