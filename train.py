import argparse
import torch
from collections import OrderedDict
from os.path import isdir
from torch import nn
from torch import optim
from torchvision import datasets, transforms, models

#default value for variables
arch = "vgg16"
learning_rate = 0.001
hidden_units = 120
epochs = 5
device = "gpu"

def arg_parser():
    parser = argparse.ArgumentParser(description="Train.py")
    parser.add_argument('--arch', dest="arch", action="store", type = str)
    parser.add_argument('--save_dir', dest="save_dir", action="store", default="./checkpoint.pth")
    parser.add_argument('--learning_rate', dest="learning_rate", action="store")
    parser.add_argument('--hidden_units', type=int, dest="hidden_units", action="store")
    parser.add_argument('--epochs', dest="epochs", action="store", type=int)
    parser.add_argument('--gpu', dest="gpu", action="store")
    args = parser.parse_args()
    if args.arch:
        arch = args.arch
    if args.learning_rate:
        learning_rate = args.learning_rate
    if args.hidden_units:
        hidden_units = args.hidden_units
    if args.epochs:
        epochs = args.epochs
    if args.gpu:
        gpu = args.gpu
    return args



def load_data(root='flowers'):
    # Set directory for training
    data_dir = root
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    #training transforms
    train_transforms = transforms.Compose([transforms.Resize(224),
                                       transforms.CenterCrop(224),
                                      transforms.RandomHorizontalFlip(0.2),
                                      transforms.RandomRotation(25),
                                      transforms.RandomVerticalFlip(0.1),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485,0.456, 0.406],[0.229, 0.224, 0.225])])
    #validation transforms
    valid_transforms = transforms.Compose([transforms.Resize(224),
                                      transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485,0.456, 0.406],[0.229, 0.224, 0.225])])
    #test transforms
    test_transforms = transforms.Compose([transforms.Resize(224),
                                      transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485,0.456, 0.406],[0.229, 0.224, 0.225])])

    #loading the train data
    train_datasets = datasets.ImageFolder(train_dir, transform = train_transforms)

    #loading the test data
    test_datasets = datasets.ImageFolder(test_dir, transform = test_transforms)
    #load validation data
    valid_datasets = datasets.ImageFolder(valid_dir, transform = valid_transforms)
    # TODO: Using the image datasets and the trainforms, define the dataloaders
    trainloader = torch.utils.data.DataLoader(train_datasets, batch_size = 64, shuffle = True)
    testloader = torch.utils.data.DataLoader(test_datasets, batch_size = 32)
    validloader = torch.utils.data.DataLoader(valid_datasets, batch_size = 32) 
    
    return trainloader, testloader, validloader
    

def construct_network(arch="vgg16",device='gpu', learning_rate = 0.001):

    if arch =="vgg16":
        model = models.vgg16(pretrained = True)
    else: 
        model = models.densenet121(pretrained = True)
    
   # model = models.vgg16(pretrained = True)   
    #freeze parameters and build model
    for param in model.parameters():
        param.requires_grad = False 
    classifier = nn.Sequential(OrderedDict([
                ('input', nn.Linear(25088, 4096)),
                ('relu1', nn.ReLU()),
                ('dropout',nn.Dropout(0.5)),
                ('hid1', nn.Linear(4096, 102)),
                ('out_act', nn.LogSoftmax(dim=1))]))
    model.classifier = classifier
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr = learning_rate)
    
    return model, criterion, optimizer , classifier



def validation(model, testloader, criterion, device):
    test_loss = 0
    accuracy = 0
    
    for ii, (inputs, labels) in enumerate(testloader):
        
        inputs, labels = inputs.to(device), labels.to(device)
        
        output = model.forward(inputs)
        test_loss += criterion(output, labels).item()
        
        ps = torch.exp(output)
        equality = (labels.data == ps.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()
    return test_loss, accuracy

    model.classifier = classifier
    return classifier

def network_trainer(model, trainloader, validloader, device, 
                    criterion, optimizer, epochs=5, print_every=20):
    
    print(f"Number of Epochs = {epochs}")
    print("Initializing training process...\n")
    
    for epoch in range(1, epochs+1):
        running_loss = 0
        model.train()
        
        for batch_idx, (inputs, labels) in enumerate(trainloader, 1):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
            running_loss += loss.item()
            
            if batch_idx % print_every == 0:
                model.eval()
                with torch.no_grad():
                    valid_loss, accuracy = validation(model, validloader, criterion)
            
                print(f"Epoch: {epoch}/{epochs} | "
                      f"Batch: {batch_idx} | "
                      f"Training Loss: {running_loss/print_every:.4f} | "
                      f"Validation Loss: {valid_loss/len(validloader):.4f} | "
                      f"Validation Accuracy: {accuracy/len(validloader):.4f}")
            
                running_loss = 0
                model.train()

    return model


def validate_model(model, testloader, device):
   # Do validation on the test set
    correct,total = 0,0
    with torch.no_grad():
        model.eval()
        for data in train_loader:
            images, labels = data
            images, labels = images.to('cuda'), labels.to('cuda')
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()

    print('Accuracy on test images is: %d%%' % (100 * correct_predictions / total))
    
    
def initial_checkpoint(model, Save_Dir, train_datasets):
       
    # Save model at checkpoint
    if type(Save_Dir) == type(None):
        print("Model checkpoint directory not specified, model will not be saved.")
    else:
        if isdir(Save_Dir):
            class_to_idx = train_datasets.class_to_idx
            model_checkpoint = {'model_arch': 'vgg16_bn',
                                'input': 25088,
                                'hid': hidden_units,
                                'out': number_cat,
                                'state_dict': model.state_dict(),
                                'model_class': class_to_idx}
            torch.save(model_checkpoint,'model_checkpoint.pth')        
            
            
        else: 
            print("Directory not found, model will not be saved.")

def main():
     
    # Get Keyword Args for Training
    args = arg_parser()
    
    trainloader, validloader, testloader = load_data(root='flowers')
    
    model, optimizer, criterion, classifier = construct_network(arch=args.arch)
       
    device = 'cuda' if torch.cuda.is_available() else "cpu";
    model.to(device);
    
    
    if type(args.learning_rate) == type(None):
        learning_rate = 0.001
        print("Learning rate specificed as 0.001")
    else: learning_rate = args.learning_rate
    
    
    print_every = 20
    batch = 0
    
    trained_model = network_trainer(model, trainloader, validloader,device, criterion, optimizer, epochs = 5,print_every=20)
    
    print("\nTraining process completed!!")
    predict
    validate_model(trained_model, testloader, device)
   
    initial_checkpoint(trained_model, args.save_dir, train_data)
if __name__ == '__main__': main()