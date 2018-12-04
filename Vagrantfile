Vagrant.configure("2") do |config|   
    config.vm.box = "ubuntu/xenial64" 
    config.vm.provider :virtualbox do |v|   
	v.customize ["modifyvm", :id, "--memory", 4096] 
    end
    config.vm.provision :shell, path: "bootstrap.sh"
end
