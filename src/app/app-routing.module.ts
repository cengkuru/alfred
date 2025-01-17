// src/app/app-routing.module.ts
import { NgModule } from '@angular/core';
import { RouterModule, Routes } from '@angular/router';
import {AiChatComponent} from "./ai-chat/ai-chat.component";

const routes: Routes = [
  { path: '', redirectTo: '/ai-chat', pathMatch: 'full' },
  // ai chat
  { path: 'ai-chat', component: AiChatComponent },

];

@NgModule({
  imports: [RouterModule.forRoot(routes)],
  exports: [RouterModule]
})
export class AppRoutingModule { }
