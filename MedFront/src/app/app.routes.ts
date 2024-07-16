import { Routes } from '@angular/router';
import { ClassificationComponent } from './classification/classification.component';
import { LoginComponent } from './login/login.component';
import { HomeComponent } from './home/home.component';

export const routes: Routes = [
  { path: '', component: HomeComponent },
  { path: 'login', component: LoginComponent },
  { path: 'classification', component: ClassificationComponent },
];
