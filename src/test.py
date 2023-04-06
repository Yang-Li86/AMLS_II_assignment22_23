def inference(model, test_loader, device):
    model.to(device)

    probs = []

    for i, images in enumerate(test_loader):
        images = images.to(device)

        with torch.no_grad():
            y_preds = model(images)

        probs.append(y_preds.to('cpu').numpy())

    probs = np.concatenate(probs)

    return probs