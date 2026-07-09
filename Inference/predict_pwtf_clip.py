def predict_pwtf_clip(

    pwtf,

    model,

    device

):

    """
    pwtf:
        (2,128,128)

    """

    tensor = torch.tensor(

        pwtf,

        dtype=torch.float32

    )


    tensor = tensor.unsqueeze(0)


    tensor = tensor.to(device)


    with torch.no_grad():

        output = model(tensor)


        probability = torch.sigmoid(

            output

        ).item()


    return probability
